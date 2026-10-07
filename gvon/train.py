"""Distil teacher labels into the local student: MiniLM embeddings -> LogisticRegression head.

Why: the frontier-model labels in data/labels/tweets.jsonl are the ground truth we can afford to
make offline; the proxy needs a model that answers in milliseconds on CPU/Metal. A linear head over
frozen sentence embeddings trains in seconds on a few thousand tweets and generalises better than
fine-tuning on so little data.

What the student can and cannot learn (honest scope):
- The teacher judged each tweet with the author's bio, account age, follower counts, the parent tweet
  and all of that author's tweets together. The student sees ONLY the post text (config
  `student_inputs`). Identity-driven nullify labels (spam_bot, shill, generic_filler,
  engagement_farming) are therefore not learnable from text; those rows are dropped from training and
  left to the blocklist. The positive class is content-intrinsic nullify (hostility, insults, FUD,
  rug/scam insinuations, entitled demands, sneering, ...). `--target all` restores the old target.
- Text is cleaned with gvon.classifier.content_text (leading @mention run and URLs removed), the same
  function the proxy uses at scoring time; rows with < 3 content chars (link/media-only) are skipped.

Evaluation (honest metrics): rows are grouped by author (authors sharing an identical cleaned text are
merged into one group) so no author is on both sides of any split. C is chosen once by a grouped 3-fold
grid (ROC-AUC is nearly flat across the grid, and one C keeps the probability scale identical between
evaluation and the shipped model). Metrics come from out-of-fold predictions of a 5-fold
StratifiedGroupKFold, i.e. they estimate performance on UNSEEN authors, which is what the proxy faces
(seen, blocklisted authors never reach the text model). Bootstrap CIs resample author groups.

Calibration: the weighted head's raw probabilities are not P(nullify). The shipped head
(gvon.classifier.CalibratedHead) adds a Platt calibrator fit, unweighted, on out-of-fold decision values;
the evaluated OOF probabilities are cross-fitted the same way (each fold calibrated on the other folds),
so thresholds are chosen and applied on one, interpretable scale (0.6 ~ "60% of such posts are nullify").

Threshold: the smallest t whose out-of-fold precision >= GVON_MIN_PRECISION (default 0.7, with at
least MIN_TP_AT_THRESHOLD true positives). An explicit --threshold or GVON_THRESHOLD overrides it
(threshold_source records which). If no t qualifies and no override is given, the config is written
with text_model_enabled=false and threshold=1.01, so the proxy filters by blocklist only, and a loud
warning is printed.

Weights: teacher confidence (0.5+|score-0.5|) x 1/sqrt(rows by that author), then rescaled so both
classes carry equal total weight (no class_weight, so one prolific account cannot dominate a class).

CLI:
    python -m gvon.train [--labels data/labels/tweets.jsonl] [--raw data/raw/tweets.jsonl]
                         [--out models/latest] [--threshold T] [--min-precision 0.7]
                         [--target content|all] [--embedding-model NAME] [--no-score-weight]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from gvon import env
from gvon.classifier import (DEFAULT_EMBEDDING_MODEL, MIN_CONTENT_CHARS, CalibratedHead, content_text, encode,
                             load_encoder, pick_device)

DEFAULT_LABELS = env.ROOT / "data" / "labels" / "tweets.jsonl"
DEFAULT_RAW = env.ROOT / "data" / "raw" / "tweets.jsonl"
DEFAULT_OUT = env.ROOT / "models" / "latest"
C_GRID: tuple[float, ...] = (0.01, 0.03, 0.1, 0.3, 1.0)
OUTER_FOLDS = 5
INNER_FOLDS = 3
SEED = 1337
VALID_LABELS = {"nullify", "neutral", "good"}
DEFAULT_MIN_PRECISION = 0.7
MIN_TP_AT_THRESHOLD = 5
DISABLED_THRESHOLD = 1.01  # > any probability: text scoring can never fire
BOOTSTRAP_ROUNDS = 500

# Nullify reasons a text-only model can see in the words themselves.
CONTENT_TAGS = frozenset({
    "hostility", "insult", "slur", "threat", "contempt", "scam_accusation", "fud", "rug_insinuation",
    "sneering", "mockery", "condescension", "doom", "baiting", "concern_trolling", "entitled_demand", "pile_on",
})
# Nullify reasons that are about WHO posted (bot/shill/farm accounts): the blocklist's job, not the text model's.
IDENTITY_TAGS = frozenset({"spam_bot", "shill", "generic_filler", "engagement_farming"})
TARGETS = ("content", "all")


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield one dict per non-blank line; a bad line is a hard error so corrupt data is not silently dropped."""
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON ({exc.msg})") from exc


def load_raw_texts(path: str | Path) -> dict[str, str]:
    """id -> text from data/raw/tweets.jsonl."""
    return {str(r["id"]): str(r.get("text") or "") for r in iter_jsonl(path) if "id" in r}


def load_labels(path: str | Path) -> dict[str, dict[str, Any]]:
    """id -> label row; later rows for the same id win (a re-label overrides)."""
    out: dict[str, dict[str, Any]] = {}
    for r in iter_jsonl(path):
        if "id" in r and r.get("label") in VALID_LABELS:
            out[str(r["id"])] = r
    return out


def confidence_weight(score: Any) -> float:
    """0.5 + |score - 0.5|: confident teacher labels count up to 2x the ambiguous ones."""
    try:
        s = float(score)
    except (TypeError, ValueError):
        return 1.0
    if not np.isfinite(s):
        return 1.0
    return 0.5 + abs(min(max(s, 0.0), 1.0) - 0.5)


def first_tag(reasons: Any) -> str:
    """Normalised first reason tag ("Rug insinuation" -> "rug_insinuation"); "" when absent."""
    if not isinstance(reasons, list) or not reasons:
        return ""
    return re.sub(r"[^a-z0-9]+", "_", str(reasons[0]).strip().lower()).strip("_")


def make_groups(authors: list[str], texts: list[str]) -> np.ndarray:
    """Group id per row: the author, with authors that posted an identical cleaned text merged (union-find),
    so copy-paste text from different accounts cannot sit on both sides of a split."""
    parent: dict[str, str] = {}

    def find(a: str) -> str:
        parent.setdefault(a, a)
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    first_author: dict[str, str] = {}
    for a, t in zip(authors, texts):
        key = " ".join(t.lower().split())
        if key in first_author:
            ra, rb = find(a), find(first_author[key])
            if ra != rb:
                parent[ra] = rb
        else:
            first_author[key] = a
            find(a)
    roots: dict[str, int] = {}
    return np.asarray([roots.setdefault(find(a), len(roots)) for a in authors], dtype=np.int64)


@dataclass
class Dataset:
    texts: list[str]  # cleaned content_text, exactly what the encoder sees
    y: np.ndarray
    w: np.ndarray
    groups: np.ndarray
    authors: list[str]
    labels: list[str]
    tags: list[str]
    counts: dict[str, int] = field(default_factory=dict)
    unlisted_content_tags: dict[str, int] = field(default_factory=dict)


def balance_weights(w: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Rescale so each class carries half the total weight (mean weight 1)."""
    w = w.astype(np.float64).copy()
    n = len(y)
    for cls in (0, 1):
        s = w[y == cls].sum()
        if s > 0:
            w[y == cls] *= (n / 2.0) / s
    return w


def build_dataset(labels: dict[str, dict[str, Any]], raw: dict[str, str], use_weights: bool = True,
                  target: str = "content") -> Dataset:
    """Join labels with text and decide each row's training target.

    target="content": nullify rows count as positive only when their first reason is about the words
    (CONTENT_TAGS, or a free-form tag outside IDENTITY_TAGS: the rubric allows new tags); identity-only
    nullify rows (IDENTITY_TAGS) are dropped, not relabelled 0. good/neutral -> 0.
    """
    if target not in TARGETS:
        raise ValueError(f"target must be one of {TARGETS}")
    texts: list[str] = []
    ys: list[int] = []
    conf: list[float] = []
    authors: list[str] = []
    labs: list[str] = []
    tags: list[str] = []
    counts: Counter[str] = Counter()
    unlisted: Counter[str] = Counter()
    for tid, row in labels.items():
        text = raw.get(tid) or row.get("text") or ""
        if not text.strip():
            counts["skipped_no_text"] += 1
            continue
        clean = content_text(text)
        if len(clean) < MIN_CONTENT_CHARS:
            counts["skipped_low_info"] += 1
            continue
        lab = row["label"]
        tag = first_tag(row.get("reasons"))
        if lab == "nullify" and target == "content":
            if tag in IDENTITY_TAGS or not tag:
                counts["skipped_identity_nullify"] += 1
                continue
            if tag not in CONTENT_TAGS:
                unlisted[tag] += 1
        counts[lab] += 1
        texts.append(clean)
        ys.append(1 if lab == "nullify" else 0)
        conf.append(confidence_weight(row.get("nullify_score")) if use_weights else 1.0)
        authors.append(str(row.get("author_id") or f"_tweet_{tid}"))
        labs.append(lab)
        tags.append(tag)
    y = np.asarray(ys, dtype=np.int64)
    per_author = Counter(authors)
    w = np.asarray([c / np.sqrt(per_author[a]) for c, a in zip(conf, authors)], dtype=np.float64)
    return Dataset(texts=texts, y=y, w=balance_weights(w, y) if len(y) else w, groups=make_groups(authors, texts),
                   authors=authors, labels=labs, tags=tags, counts=dict(counts), unlisted_content_tags=dict(unlisted))


def make_head(c: float) -> Any:
    """Plain L2 logistic regression; class balance comes from the sample weights, not class_weight."""
    from sklearn.linear_model import LogisticRegression

    return LogisticRegression(C=c, max_iter=2000)


def grouped_folds(y: np.ndarray, groups: np.ndarray, n_splits: int) -> list[tuple[np.ndarray, np.ndarray]] | None:
    """StratifiedGroupKFold splits (no group on both sides), shrinking k to what the data allows; None if < 2."""
    from sklearn.model_selection import StratifiedGroupKFold

    k = min(n_splits, len(set(groups[y == 1].tolist())), len(set(groups[y == 0].tolist())))
    if k < 2:
        return None
    sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=SEED)
    return [(tr, va) for tr, va in sgkf.split(np.zeros(len(y)), y, groups)]


def choose_c(x: np.ndarray, y: np.ndarray, w: np.ndarray, groups: np.ndarray) -> tuple[float, dict[str, float | None]]:
    """Grouped 3-fold CV over C_GRID scored by ROC-AUC; C=1 when the data is too small to split."""
    from sklearn.metrics import roc_auc_score

    folds = grouped_folds(y, groups, INNER_FOLDS)
    if folds is None:
        return 1.0, {str(c): None for c in C_GRID}
    folds = [(tr, va) for tr, va in folds if len(set(y[tr].tolist())) == 2 and len(set(y[va].tolist())) == 2]
    if not folds:
        return 1.0, {str(c): None for c in C_GRID}
    results: dict[str, float | None] = {}
    best_c, best_auc = 1.0, -1.0
    for c in C_GRID:
        aucs = [roc_auc_score(y[va], make_head(c).fit(x[tr], y[tr], sample_weight=w[tr]).predict_proba(x[va])[:, 1])
                for tr, va in folds]
        mean = float(np.mean(aucs))
        results[str(c)] = round(mean, 4)
        if mean > best_auc + 1e-9:
            best_c, best_auc = c, mean
    return best_c, results


def fit_platt(d: np.ndarray, y: np.ndarray) -> Any:
    """1-D logistic calibrator on decision values, unweighted so it learns the real class prevalence."""
    from sklearn.linear_model import LogisticRegression

    return LogisticRegression(C=100.0, max_iter=1000).fit(np.asarray(d).reshape(-1, 1), y)


def oof_predictions(x: np.ndarray, y: np.ndarray, w: np.ndarray, groups: np.ndarray, c: float
                    ) -> tuple[np.ndarray, np.ndarray] | None:
    """(OOF decision values, cross-fitted calibrated OOF P(nullify)) from grouped outer folds with a fixed C.

    Each fold's probabilities use a calibrator fit on the OTHER folds' decision values, so neither the
    head nor the calibrator has seen the author being scored."""
    folds = grouped_folds(y, groups, OUTER_FOLDS)
    if folds is None:
        return None
    d = np.full(len(y), np.nan)
    fold_of = np.full(len(y), -1)
    for k, (tr, va) in enumerate(folds):
        if len(set(y[tr].tolist())) < 2:
            continue
        d[va] = make_head(c).fit(x[tr], y[tr], sample_weight=w[tr]).decision_function(x[va])
        fold_of[va] = k
    p = np.full(len(y), np.nan)
    for k in range(len(folds)):
        va = fold_of == k
        other = (fold_of >= 0) & ~va
        if va.any() and len(set(y[other].tolist())) == 2:
            p[va] = fit_platt(d[other], y[other]).predict_proba(d[va].reshape(-1, 1))[:, 1]
    return d, p


def select_threshold(y: np.ndarray, p: np.ndarray, min_precision: float, min_tp: int = MIN_TP_AT_THRESHOLD) -> float | None:
    """Smallest threshold (highest recall) whose precision >= min_precision with >= min_tp true positives."""
    from sklearn.metrics import precision_recall_curve

    if y.sum() == 0:
        return None
    prec, rec, thr = precision_recall_curve(y, p)
    npos = int(y.sum())
    for i, t in enumerate(thr):  # ascending thresholds: the first hit is the smallest
        if prec[i] >= min_precision and round(rec[i] * npos) >= min_tp:
            return float(t)
    return None


def at_threshold(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, Any]:
    from sklearn.metrics import f1_score, precision_score, recall_score

    pred = (p >= threshold).astype(int)
    return {
        "threshold": round(float(threshold), 4),
        "precision": round(float(precision_score(y, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y, pred, zero_division=0)), 4),
        "flagged_frac": round(float(pred.mean()), 4) if len(pred) else None,
        "true_positives": int((pred & y).sum()),
    }


def bootstrap_ci(y: np.ndarray, p: np.ndarray, groups: np.ndarray, threshold: float | None,
                 rounds: int = BOOTSTRAP_ROUNDS) -> dict[str, list[float] | None]:
    """95% percentile CIs from resampling whole author groups (rows of one author are correlated)."""
    from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

    rng = np.random.default_rng(SEED)
    by_group: dict[int, list[int]] = defaultdict(list)
    for i, g in enumerate(groups.tolist()):
        by_group[g].append(i)
    keys = list(by_group)
    stats: dict[str, list[float]] = {"roc_auc": [], "average_precision": [], "f1": []}
    for _ in range(rounds):
        idx = np.concatenate([by_group[keys[k]] for k in rng.integers(0, len(keys), len(keys))])
        yb, pb = y[idx], p[idx]
        if len(set(yb.tolist())) < 2:
            continue
        stats["roc_auc"].append(roc_auc_score(yb, pb))
        stats["average_precision"].append(average_precision_score(yb, pb))
        if threshold is not None:
            stats["f1"].append(f1_score(yb, (pb >= threshold).astype(int), zero_division=0))
    return {k: ([round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)] if v else None)
            for k, v in stats.items()}


def oof_metrics(ds: Dataset, p: np.ndarray, threshold: float | None, oof_threshold: float | None) -> dict[str, Any]:
    """Unseen-author metrics from out-of-fold predictions (rows without a prediction are excluded)."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    ok = ~np.isnan(p)
    y, pp, groups = ds.y[ok], p[ok], ds.groups[ok]
    tags = [t for t, keep in zip(ds.tags, ok) if keep]
    labs = [lab for lab, keep in zip(ds.labels, ok) if keep]
    two = len(set(y.tolist())) == 2
    m: dict[str, Any] = {
        "split": "grouped_by_author",
        "evaluation": "out-of-fold predictions, StratifiedGroupKFold(5) outer x grouped 3-fold C grid inner",
        "n_eval": int(ok.sum()),
        "n_eval_nullify": int(y.sum()),
        "base_rate": round(float(y.mean()), 4) if len(y) else None,
        "roc_auc": round(float(roc_auc_score(y, pp)), 4) if two else None,
        "average_precision": round(float(average_precision_score(y, pp)), 4) if two else None,
        "mean_p_negatives": round(float(pp[y == 0].mean()), 4) if (y == 0).any() else None,
        "mean_p_positives": round(float(pp[y == 1].mean()), 4) if (y == 1).any() else None,
    }
    if oof_threshold is not None:
        m["at_oof_threshold"] = at_threshold(y, pp, oof_threshold)
        m["at_oof_threshold"]["note"] = "threshold selected on these same OOF rows: optimistic"
    if threshold is not None and threshold <= 1.0:
        m["at_operative_threshold"] = at_threshold(y, pp, threshold)
        pred = pp >= threshold
        m["recall_by_reason"] = {
            tag: {"n": int(n), "recalled": int(sum(1 for t, yy, pr in zip(tags, y, pred) if t == tag and yy and pr))}
            for tag, n in Counter(t for t, yy in zip(tags, y) if yy).most_common()}
        m["flag_rate_by_label"] = {
            lab: round(float(np.mean([pr for la, pr in zip(labs, pred) if la == lab])), 4)
            for lab in sorted(set(labs))}
    m["ci95"] = bootstrap_ci(y, pp, groups, threshold if threshold is not None and threshold <= 1.0 else None) if two else None
    return m


def atomic_write(path: Path, write: Any) -> None:
    """Write via a temp file + rename so a running proxy never reads a half-written model."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    os.close(fd)
    try:
        write(tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def env_threshold() -> float | None:
    raw = (env.get("GVON_THRESHOLD") or "").strip()
    return float(raw) if raw else None


def min_precision_default() -> float:
    return float((env.get("GVON_MIN_PRECISION") or "").strip() or DEFAULT_MIN_PRECISION)


def decide_threshold(cli: float | None, env_thr: float | None, oof_thr: float | None) -> tuple[float, str, bool]:
    """(threshold, threshold_source, text_model_enabled): explicit overrides win, else the OOF gate."""
    if cli is not None:
        return float(cli), "cli", True
    if env_thr is not None:
        return float(env_thr), "env", True
    if oof_thr is not None:
        return round(oof_thr, 6), "oof_precision_target", True
    return DISABLED_THRESHOLD, "none_met_precision_target", False


def train(
    labels_path: str | Path = DEFAULT_LABELS,
    raw_path: str | Path = DEFAULT_RAW,
    out_dir: str | Path = DEFAULT_OUT,
    threshold: float | None = None,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    use_weights: bool = True,
    min_precision: float | None = None,
    target: str = "content",
) -> dict[str, Any]:
    """Train and write head.joblib + config.json; returns the config dict."""
    import joblib

    min_precision = min_precision_default() if min_precision is None else float(min_precision)
    ds = build_dataset(load_labels(labels_path), load_raw_texts(raw_path), use_weights, target)
    if len(set(ds.y.tolist())) < 2:
        raise ValueError(f"need both nullify and non-nullify examples with text; got {ds.counts}")

    device = pick_device()
    x = encode(load_encoder(embedding_model, device), ds.texts)

    best_c, grid = choose_c(x, ds.y, ds.w, ds.groups)
    oof = oof_predictions(x, ds.y, ds.w, ds.groups, best_c)
    oof_thr = None
    if oof is not None:
        d, p = oof
        ok = ~np.isnan(p)
        oof_thr = select_threshold(ds.y[ok], p[ok], min_precision)
    thr, source, enabled = decide_threshold(threshold, env_threshold(), oof_thr)

    metrics: dict[str, Any]
    if oof is None:
        metrics = {"split": "grouped_by_author", "note": "too few author groups per class for grouped CV; metrics unavailable"}
    else:
        metrics = oof_metrics(ds, p, thr if enabled else None, oof_thr)
    metrics["C"] = best_c
    metrics["C_selection"] = "grouped 3-fold ROC-AUC on all rows, chosen once (slightly optimistic; grid is near-flat)"
    metrics["cv_roc_auc_by_C"] = grid
    metrics["sample_weighting"] = (("0.5+|nullify_score-0.5|" if use_weights else "1") +
                                   " x 1/sqrt(rows per author), classes rescaled to equal total weight")

    base = make_head(best_c).fit(x, ds.y, sample_weight=ds.w)
    calibrated = False
    final: Any = base
    if oof is not None:
        okd = ~np.isnan(d)
        if len(set(ds.y[okd].tolist())) == 2:
            final, calibrated = CalibratedHead(base, fit_platt(d[okd], ds.y[okd])), True
    metrics["calibration"] = "platt on out-of-fold decision values (unweighted)" if calibrated else "none (too little data)"
    label_authors = {lab: len({a for a, la in zip(ds.authors, ds.labels) if la == lab}) for lab in sorted(set(ds.labels))}
    top = Counter(a for a, yy in zip(ds.authors, ds.y.tolist()) if yy).most_common(1)
    config = {
        "embedding_model": embedding_model,
        "threshold": thr,
        "threshold_source": source,
        "text_model_enabled": enabled,
        "min_precision": min_precision,
        "oof_threshold": round(oof_thr, 6) if oof_thr is not None else None,
        "oof_precision_target_met": oof_thr is not None,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_train": int(len(ds.y)),
        "n_authors": int(len(set(ds.authors))),
        "student_inputs": ["tweet_text"],
        "probability_calibrated": calibrated,
        "text_preprocessing": "gvon.classifier.content_text (leading @mentions and URLs removed); < 3 chars skipped",
        "target": target if target == "all" else "content: nullify with a content reason tag; identity-only nullify dropped",
        "metrics": metrics,
        "label_counts": ds.counts,
        "label_authors": label_authors,
        "top_positive_author_share": round(top[0][1] / max(1, int(ds.y.sum())), 4) if top else None,
        "unlisted_content_tags": ds.unlisted_content_tags,
        "train_device": device,
    }
    out = Path(out_dir)
    atomic_write(out / "head.joblib", lambda path: joblib.dump(final, path))
    atomic_write(out / "config.json", lambda path: Path(path).write_text(json.dumps(config, indent=2) + "\n"))
    return config


def warn_if_gated(cfg: dict[str, Any]) -> None:
    """Make an unmet precision target impossible to miss."""
    if cfg["oof_precision_target_met"]:
        return
    bar = "!" * 78
    msg = (f"{bar}\nWARNING: no threshold reached out-of-fold precision >= {cfg['min_precision']} "
           f"(with >= {MIN_TP_AT_THRESHOLD} true positives) on unseen authors.\n")
    if cfg["text_model_enabled"]:
        msg += (f"Text scoring stays ON only because threshold {cfg['threshold']} was given explicitly "
                f"({cfg['threshold_source']}); expect many wrong hides. Unset GVON_THRESHOLD to let the gate decide.\n")
    else:
        msg += "Text scoring is DISABLED (text_model_enabled=false): the proxy filters by blocklist only.\n"
    print(msg + bar, file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m gvon.train", description=__doc__.splitlines()[0])
    ap.add_argument("--labels", default=str(DEFAULT_LABELS))
    ap.add_argument("--raw", default=str(DEFAULT_RAW))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--threshold", type=float, default=None,
                    help="explicit operative threshold (default: GVON_THRESHOLD, else the OOF precision gate)")
    ap.add_argument("--min-precision", type=float, default=None,
                    help=f"OOF precision target for the learned threshold (default: GVON_MIN_PRECISION or {DEFAULT_MIN_PRECISION})")
    ap.add_argument("--target", choices=TARGETS, default="content",
                    help="content (default): identity-only nullify rows dropped; all: every nullify row is positive")
    ap.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    ap.add_argument("--no-score-weight", action="store_true", help="disable nullify_score confidence weighting")
    args = ap.parse_args(argv)
    cfg = train(args.labels, args.raw, args.out, args.threshold, args.embedding_model, not args.no_score_weight,
                args.min_precision, args.target)
    keys = ("n_train", "n_authors", "threshold", "threshold_source", "text_model_enabled", "oof_threshold",
            "label_counts", "label_authors", "metrics")
    print(json.dumps({k: cfg[k] for k in keys}, indent=2))
    warn_if_gated(cfg)
    print(f"wrote {Path(args.out) / 'head.joblib'} and {Path(args.out) / 'config.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
