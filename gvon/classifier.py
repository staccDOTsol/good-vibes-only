"""Local, fast "student" classifier: sentence embeddings + a logistic-regression head.

Why this shape: a frontier model labels engagements offline (expensive, slow, high quality); this
module distils those labels into something that can sit in the network path of x.com responses.
MiniLM-L6 embeddings (~22M params) + a linear head run in milliseconds on CPU or Apple Metal (MPS),
so the proxy can score a timeline page without noticeable latency.

Public API (see CONTRACT.md):
    Nullifier(model_dir, blocklist_path, threshold, watchlist_path).score(texts) -> list[float]   # P(nullify)
    Nullifier(...).should_nullify(text, author_id=..., username=...) -> bool

Inputs: the student sees ONLY the post text, cleaned by content_text() (leading @mention run and URLs
removed, exactly as in gvon.train). The teacher judged with bio, account age, follower counts and the
parent tweet; identity-driven cases (bots, shills, copy-paste spam) are the blocklist's job, not this
model's. Texts with < MIN_CONTENT_CHARS of content (link-only / media-only replies) score 0.0: the model
abstains and leaves them to the blocklist.

Threshold: one resolution order everywhere (proxy, CLI, tests): explicit argument > GVON_THRESHOLD >
config.json > 0.5 (resolve_threshold). If training found no threshold meeting its precision target it
writes text_model_enabled=false; text scoring is then off unless a threshold is given explicitly.

Per-author signals: accounts in data/watchlist.json (teacher verdict "watch") get a lower threshold
(threshold - GVON_WATCH_DELTA, default 0.15), and a bounded running mean of each author's distinct scored
texts nullifies an author once n >= AUTHOR_AGG_MIN_N and mean P >= threshold.

CLI:
    python -m gvon.classifier "text" ["text2" ...]   # print P(nullify) per text
    python -m gvon.classifier --bench                # time load + a 50-text batch, print device + ms
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sys
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np

from gvon import env

log = logging.getLogger("gvon.classifier")

# Anchored at the repo root (where .env lives), not the cwd, so running from another directory never
# reads or writes real data outside this repo's gitignored data/ and models/.
DEFAULT_MODEL_DIR = env.ROOT / "models" / "latest"
DEFAULT_BLOCKLIST = env.ROOT / "data" / "blocklist.json"
WATCHLIST_NAME = "watchlist.json"  # written by gvon.label next to blocklist.json
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CACHE_MAX = 100_000
ENCODE_BATCH = 64
MIN_CONTENT_CHARS = 3
DEFAULT_WATCH_DELTA = 0.15
AUTHOR_AGG_MIN_N = 3
AUTHOR_AGG_MAX_AUTHORS = 20_000
AUTHOR_AGG_MAX_TEXTS = 50

_LEADING_MENTIONS = re.compile(r"^(?:\s*@\w+)+")
_URL = re.compile(r"https?://\S+")


def content_text(text: str | None) -> str:
    """What the student actually reads: the post minus its leading @mention run and URLs.

    Reply text from the X API starts with the @handles being replied to (nearly always the owner's), a
    reply-target cue the home timeline lacks; t.co links carry no meaning. Shared by train and score so
    both sides see identical input.
    """
    t = _LEADING_MENTIONS.sub("", text or "")
    return " ".join(_URL.sub(" ", t).split())


def is_low_info(text: str | None) -> bool:
    """Link-only / media-only / mention-only posts: nothing for a text model to judge."""
    return len(content_text(text)) < MIN_CONTENT_CHARS


def resolve_threshold(explicit: float | None, config: dict[str, Any]) -> tuple[float, str]:
    """(threshold, source) in the one order every entry point uses: arg > GVON_THRESHOLD > config > 0.5."""
    if explicit is not None:
        return float(explicit), "argument"
    env_thr = (env.get("GVON_THRESHOLD") or "").strip()
    if env_thr:
        return float(env_thr), "env"
    if config.get("threshold") is not None:
        return float(config["threshold"]), "config"
    return 0.5, "default"


def pick_device() -> str:
    """Return the torch device to embed on.

    GVON_DEVICE (env or .env) wins so a user can force "cpu" when MPS is slower for tiny batches
    or misbehaves; otherwise prefer Apple Metal when present.
    """
    forced = env.get("GVON_DEVICE")
    if forced:
        return forced.strip()
    import torch

    return "mps" if torch.backends.mps.is_available() else "cpu"


def load_encoder(model_name: str, device: str) -> Any:
    """Load a SentenceTransformer, imported lazily because importing torch/transformers costs seconds.

    Tries the local Hugging Face cache first (local_files_only) so a warm start makes no network
    round-trips (those cost ~2s); only the very first run falls through to a download.
    """
    from sentence_transformers import SentenceTransformer

    try:
        from transformers.utils import logging as hf_logging

        hf_logging.disable_progress_bar()
    except Exception:  # cosmetic only
        pass
    try:
        return SentenceTransformer(model_name, device=device, local_files_only=True)
    except Exception:
        return SentenceTransformer(model_name, device=device)


def encode(encoder: Any, texts: list[str], batch_size: int = ENCODE_BATCH) -> np.ndarray:
    """Embed texts as L2-normalised float32 vectors (normalisation keeps the linear head scale-free)."""
    if not texts:
        dim = encoder.get_sentence_embedding_dimension() or 0
        return np.zeros((0, dim), dtype=np.float32)
    vecs = encoder.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return np.asarray(vecs, dtype=np.float32)


class CalibratedHead:
    """Linear head + Platt calibrator, pickled as models/latest/head.joblib by gvon.train.

    Why: the head is trained with class-balanced sample weights, so its raw predict_proba is not a real
    P(nullify) and its scale moves with every retrain. The calibrator is a 1-D logistic fit (unweighted)
    on out-of-fold decision values, so predict_proba estimates the real nullify rate and a threshold such
    as 0.6 keeps its meaning across retrains.
    """

    def __init__(self, base: Any, calibrator: Any) -> None:
        self.base = base
        self.calibrator = calibrator
        self.classes_ = getattr(calibrator, "classes_", getattr(base, "classes_", np.asarray([0, 1])))

    def decision_function(self, x: np.ndarray) -> np.ndarray:
        return self.base.decision_function(x)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return self.calibrator.predict_proba(np.asarray(self.base.decision_function(x)).reshape(-1, 1))


def text_key(text: str) -> str:
    """Cache key: sha1 of the UTF-8 text, so the cache does not hold raw tweet text as keys."""
    return hashlib.sha1(text.encode("utf-8", "surrogatepass")).hexdigest()


def normalize_username(username: str | None) -> str:
    """Usernames compare case-insensitively and without a leading '@'."""
    return (username or "").strip().lstrip("@").strip().lower()


def load_blocklist(path: str | Path) -> tuple[set[str], set[str]]:
    """Read data/blocklist.json (or watchlist.json, same shape) into (ids, usernames).

    A missing file means "nobody listed", not an error."""
    p = Path(path)
    if not p.exists():
        return set(), set()
    data = json.loads(p.read_text() or "{}")
    ids: set[str] = set()
    names: set[str] = set()
    for acct in data.get("accounts", []) or []:
        if acct.get("id") not in (None, ""):
            ids.add(str(acct["id"]).strip())
        name = normalize_username(acct.get("username"))
        if name:
            names.add(name)
    return ids, names


def read_config(model_dir: str | Path) -> dict[str, Any]:
    """Load models/<dir>/config.json written by gvon.train."""
    cfg_path = Path(model_dir) / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"{cfg_path} not found; run `python -m gvon.train` first")
    return json.loads(cfg_path.read_text())


class Nullifier:
    """Scores text for P(nullify) and decides whether to hide it.

    The embedding model is loaded lazily on the first score() so constructing a Nullifier (e.g. at
    proxy start-up) is cheap and blocklist-only decisions never pay for torch.
    """

    def __init__(
        self,
        model_dir: str | Path = DEFAULT_MODEL_DIR,
        blocklist_path: str | Path = DEFAULT_BLOCKLIST,
        threshold: float | None = None,
        watchlist_path: str | Path | None = None,
    ) -> None:
        import joblib

        self.model_dir = Path(model_dir)
        self.config = read_config(self.model_dir)
        self.head = joblib.load(self.model_dir / "head.joblib")
        self.embedding_model: str = self.config.get("embedding_model", DEFAULT_EMBEDDING_MODEL)
        self.threshold, self.threshold_source = resolve_threshold(threshold, self.config)
        # An explicit threshold (argument or GVON_THRESHOLD) is the user's override and re-enables text
        # scoring even when training found no threshold meeting its precision target.
        self.text_model_enabled: bool = (self.threshold_source in ("argument", "env")
                                         or bool(self.config.get("text_model_enabled", True)))
        self.watch_delta = float((env.get("GVON_WATCH_DELTA") or "").strip() or DEFAULT_WATCH_DELTA)
        self.blocklist_path = Path(blocklist_path)
        self.watchlist_path = Path(watchlist_path) if watchlist_path is not None else self.blocklist_path.parent / WATCHLIST_NAME
        self.blocked_ids, self.blocked_usernames = load_blocklist(self.blocklist_path)
        self.watch_ids, self.watch_usernames = load_blocklist(self.watchlist_path)
        self.device: str = pick_device()
        self._encoder: Any = None
        self._cache: OrderedDict[str, float] = OrderedDict()
        self._authors: OrderedDict[str, OrderedDict[str, float]] = OrderedDict()
        self._lock = threading.Lock()
        self._pos_col = self._positive_column()

    def _positive_column(self) -> int:
        """Column of predict_proba for label 1 (nullify); guards against a head trained with odd classes_."""
        classes = list(getattr(self.head, "classes_", [0, 1]))
        return classes.index(1) if 1 in classes else len(classes) - 1

    def reload_blocklist(self) -> None:
        """Re-read the block/watch lists (the labeler may regenerate them while the proxy runs)."""
        self.blocked_ids, self.blocked_usernames = load_blocklist(self.blocklist_path)
        self.watch_ids, self.watch_usernames = load_blocklist(self.watchlist_path)

    def _ensure_encoder(self) -> Any:
        if self._encoder is None:
            self._encoder = load_encoder(self.embedding_model, self.device)
        return self._encoder

    def warmup(self) -> None:
        """Load the encoder and run one tiny batch so the first real request is not slow."""
        with self._lock:
            encode(self._ensure_encoder(), ["warmup"])

    def _cache_put(self, key: str, value: float) -> None:
        self._cache[key] = value
        self._cache.move_to_end(key)
        while len(self._cache) > CACHE_MAX:
            self._cache.popitem(last=False)

    def score(self, texts: list[str]) -> list[float]:
        """P(nullify) for each text (0.0 = abstain for low-info texts). Batched; texts that are identical
        after content_text() cleaning are scored once and cached by sha1."""
        cleaned = [content_text(str(t)) for t in texts]
        keys = [text_key(c) for c in cleaned]
        with self._lock:
            missing: dict[str, str] = {}
            for k, c in zip(keys, cleaned):
                if k in self._cache:
                    self._cache.move_to_end(k)
                elif len(c) < MIN_CONTENT_CHARS:
                    self._cache_put(k, 0.0)
                elif k not in missing:
                    missing[k] = c
            if missing:
                vecs = encode(self._ensure_encoder(), list(missing.values()))
                probs = self.head.predict_proba(vecs)[:, self._pos_col]
                for k, p in zip(missing.keys(), probs):
                    self._cache_put(k, float(p))
            return [self._cache[k] for k in keys]

    def is_blocked(self, *, author_id: str | None = None, username: str | None = None) -> bool:
        """True when the author is on the blocklist by id or (case-insensitive, '@'-stripped) username."""
        if author_id not in (None, "") and str(author_id).strip() in self.blocked_ids:
            return True
        name = normalize_username(username)
        return bool(name) and name in self.blocked_usernames

    def is_watched(self, *, author_id: str | None = None, username: str | None = None) -> bool:
        """True for teacher-verdict "watch" accounts (data/watchlist.json)."""
        if author_id not in (None, "") and str(author_id).strip() in self.watch_ids:
            return True
        name = normalize_username(username)
        return bool(name) and name in self.watch_usernames

    def threshold_for(self, *, author_id: str | None = None, username: str | None = None) -> float:
        """Watch-listed authors get a lower bar (threshold - GVON_WATCH_DELTA)."""
        if self.is_watched(author_id=author_id, username=username):
            return self.threshold - self.watch_delta
        return self.threshold

    def _author_mean(self, author: str, text: str, p: float) -> tuple[float, int]:
        """Record p for this author's distinct text; return (mean P, n distinct texts). Bounded LRU."""
        with self._lock:
            texts = self._authors.get(author)
            if texts is None:
                texts = self._authors[author] = OrderedDict()
            self._authors.move_to_end(author)
            texts[text_key(content_text(text))] = p
            texts.move_to_end(text_key(content_text(text)))
            while len(texts) > AUTHOR_AGG_MAX_TEXTS:
                texts.popitem(last=False)
            while len(self._authors) > AUTHOR_AGG_MAX_AUTHORS:
                self._authors.popitem(last=False)
            return float(sum(texts.values()) / len(texts)), len(texts)

    def should_nullify(self, text: str, *, author_id: str | None = None, username: str | None = None) -> bool:
        """Blocklist first (no model call), then the text model: score >= the author's threshold, or the
        author's running mean over >= AUTHOR_AGG_MIN_N distinct texts >= threshold. Low-info texts abstain."""
        if self.is_blocked(author_id=author_id, username=username):
            return True
        if not self.text_model_enabled or is_low_info(text):
            return False
        p = self.score([text])[0]
        author = str(author_id).strip() if author_id not in (None, "") else normalize_username(username)
        # record every scored post (including ones that cross the threshold alone) before deciding
        mean, n = self._author_mean(author, text, p) if author else (p, 1)
        if p >= self.threshold_for(author_id=author_id, username=username):
            return True
        if n >= AUTHOR_AGG_MIN_N and mean >= self.threshold:
            log.info("gvon: author aggregate: %s mean P=%.3f over %d posts >= %.3f", username or author, mean, n,
                     self.threshold)
            return True
        return False


def bench_texts(n: int = 50) -> list[str]:
    """Synthetic, clearly-fake benchmark strings (unique so the cache cannot short-circuit the timing)."""
    return [f"benchmark sentence number {i} about nothing in particular, run {time.time_ns()}" for i in range(n)]


def run_bench(model_dir: str | Path, blocklist_path: str | Path) -> dict[str, Any]:
    """Time: construct + encoder load + first encode, then a cold 50-text batch, then a cached repeat."""
    t0 = time.perf_counter()
    nul = Nullifier(model_dir, blocklist_path)
    t_init = time.perf_counter()
    nul.warmup()
    t_load = time.perf_counter()
    texts = bench_texts(50)
    nul.score(texts)
    t_batch = time.perf_counter()
    nul.score(texts)
    t_cached = time.perf_counter()
    texts2 = bench_texts(50)
    t2 = time.perf_counter()
    nul.score(texts2)
    t_batch2 = time.perf_counter()
    return {
        "device": nul.device,
        "embedding_model": nul.embedding_model,
        "init_ms": round((t_init - t0) * 1000, 1),
        "load_incl_warmup_ms": round((t_load - t0) * 1000, 1),
        "batch50_first_ms": round((t_batch - t_load) * 1000, 1),
        "batch50_second_ms": round((t_batch2 - t2) * 1000, 1),
        "batch50_cached_ms": round((t_cached - t_batch) * 1000, 2),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m gvon.classifier", description=__doc__.splitlines()[0])
    ap.add_argument("texts", nargs="*", help="texts to score")
    ap.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    ap.add_argument("--blocklist", default=str(DEFAULT_BLOCKLIST))
    ap.add_argument("--threshold", type=float, default=None,
                    help="override the threshold (default: GVON_THRESHOLD, else config.json, else 0.5)")
    ap.add_argument("--device", default=None, help="force torch device (sets GVON_DEVICE)")
    ap.add_argument("--bench", action="store_true", help="time model load and a 50-text batch")
    args = ap.parse_args(argv)
    if args.device:
        os.environ["GVON_DEVICE"] = args.device
    if args.bench:
        print(json.dumps(run_bench(args.model_dir, args.blocklist), indent=2))
        return 0
    if not args.texts:
        ap.error("give one or more texts, or --bench")
    nul = Nullifier(args.model_dir, args.blocklist, args.threshold)
    print(f"# threshold={nul.threshold} (source={nul.threshold_source}) text_model_enabled={nul.text_model_enabled}",
          file=sys.stderr)
    for text, p in zip(args.texts, nul.score(args.texts)):
        flag = "NULLIFY" if nul.text_model_enabled and p >= nul.threshold else "keep"
        if is_low_info(text):
            flag += " (abstain: low-info text)"
        print(f"{p:.4f}\t{flag}\t{text}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
