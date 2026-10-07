"""Tests for gvon.train + gvon.classifier on a clearly synthetic dataset built in tmp_path.

Nothing here is real tweet data and nothing is written under data/ or models/.
The first run downloads sentence-transformers/all-MiniLM-L6-v2 (~90MB) from Hugging Face.
"""
from __future__ import annotations

import itertools
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from gvon import classifier as clf_mod
from gvon import train as train_mod
from gvon.classifier import Nullifier

HOSTILE_OPENERS = ["you are a pathetic idiot", "shut up you worthless loser", "nobody likes you, clown",
                   "you're a disgusting fraud", "go away you braindead moron", "what a stupid hateful troll"]
HOSTILE_TAILS = ["and everyone hates you", "delete your account", "you deserve nothing",
                 "you should be ashamed", "you are garbage", "log off forever"]
SUPPORT_OPENERS = ["this is wonderful, thank you", "great work, really inspiring", "love this, so helpful",
                   "congrats on the launch, well deserved", "thanks for sharing this kind idea", "beautiful thread"]
SUPPORT_TAILS = ["keep it up", "you made my day", "appreciate you", "learned a lot", "so happy for you",
                 "brilliant stuff"]


def synthetic_rows() -> tuple[list[dict], list[dict]]:
    """Build raw + label rows (CONTRACT shapes) from templated, obviously fake text."""
    raw, labels = [], []
    pairs = [(f"{a}, {b}", "nullify", 0.95) for a, b in itertools.product(HOSTILE_OPENERS, HOSTILE_TAILS)]
    pairs += [(f"{a}, {b}", "good", 0.03) for a, b in itertools.product(SUPPORT_OPENERS, SUPPORT_TAILS)]
    pairs += [(f"synthetic neutral note number {i} about the weather", "neutral", 0.2) for i in range(10)]
    tag = {"nullify": "hostility", "good": "support", "neutral": "off_topic"}
    for i, (text, label, score) in enumerate(pairs):
        tid = f"synth{i}"
        raw.append({"id": tid, "text": text, "author_id": f"a{i}", "author_username": f"synth_user_{i}",
                    "kind": "reply"})
        labels.append({"id": tid, "author_id": f"a{i}", "author_username": f"synth_user_{i}", "label": label,
                       "nullify_score": score, "reasons": [tag[label], "synthetic"]})
    return raw, labels


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


@pytest.fixture(autouse=True)
def _no_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    """The repo .env may set GVON_THRESHOLD; an empty value means "unset" for every resolver."""
    for k in ("GVON_THRESHOLD", "GVON_MIN_PRECISION", "GVON_WATCH_DELTA"):
        monkeypatch.setenv(k, "")


@pytest.fixture(scope="module")
def trained(tmp_path_factory: pytest.TempPathFactory) -> dict:
    root = tmp_path_factory.mktemp("gvon")
    raw, labels = synthetic_rows()
    raw_path, labels_path = root / "data/raw/tweets.jsonl", root / "data/labels/tweets.jsonl"
    write_jsonl(raw_path, raw)
    write_jsonl(labels_path, labels)
    blocklist = root / "data/blocklist.json"
    blocklist.write_text(json.dumps({"generated_at": "2026-01-01T00:00:00+00:00", "handle": "synthetic",
                                     "accounts": [{"id": "999", "username": "Blocked_Synth",
                                                   "nullify_score": 0.99, "summary": "synthetic"}]}))
    out = root / "models/latest"
    mp = pytest.MonkeyPatch()
    mp.setenv("GVON_THRESHOLD", "")
    rc = train_mod.main(["--labels", str(labels_path), "--raw", str(raw_path), "--out", str(out),
                         "--threshold", "0.5"])
    mp.undo()
    assert rc == 0
    return {"root": root, "model_dir": out, "blocklist": blocklist}


def test_config_written_per_contract(trained: dict) -> None:
    cfg = json.loads((trained["model_dir"] / "config.json").read_text())
    assert (trained["model_dir"] / "head.joblib").exists()
    for key in ("embedding_model", "threshold", "trained_at", "n_train", "metrics", "label_counts"):
        assert key in cfg
    assert cfg["embedding_model"] == clf_mod.DEFAULT_EMBEDDING_MODEL
    assert cfg["threshold"] == 0.5 and cfg["threshold_source"] == "cli" and cfg["text_model_enabled"] is True
    assert cfg["n_train"] == 36 + 36 + 10
    assert cfg["label_counts"] == {"nullify": 36, "good": 36, "neutral": 10}
    assert cfg["student_inputs"] == ["tweet_text"] and cfg["probability_calibrated"] is True
    assert cfg["oof_precision_target_met"] is True and cfg["oof_threshold"] is not None
    m = cfg["metrics"]
    assert m["split"] == "grouped_by_author"
    for key in ("roc_auc", "average_precision", "at_operative_threshold", "recall_by_reason", "ci95", "C"):
        assert key in m
    assert "best_f1_threshold" not in m
    assert m["C"] in train_mod.C_GRID
    assert m["roc_auc"] >= 0.9


def test_hostile_scores_higher_than_supportive(trained: dict) -> None:
    nul = Nullifier(trained["model_dir"], trained["blocklist"])
    hostile = ["you absolute idiot, everyone hates you", "shut up loser, delete your account"]
    supportive = ["thank you, this was wonderful and helpful", "congrats, so happy for you"]
    h, s = nul.score(hostile), nul.score(supportive)
    assert all(0.0 <= p <= 1.0 for p in h + s)
    assert min(h) > max(s)
    assert nul.should_nullify(hostile[0])
    assert not nul.should_nullify(supportive[0])


def test_blocklist_override(trained: dict) -> None:
    nul = Nullifier(trained["model_dir"], trained["blocklist"])
    kind = "thank you, this was wonderful and helpful"
    assert not nul.should_nullify(kind)
    assert nul.should_nullify(kind, author_id="999")
    assert nul.should_nullify(kind, username="@blocked_SYNTH")
    assert nul.should_nullify(kind, username="BLOCKED_SYNTH")
    assert not nul.should_nullify(kind, author_id="1000", username="someone_else")


def test_missing_blocklist_is_empty(trained: dict, tmp_path: Path) -> None:
    nul = Nullifier(trained["model_dir"], tmp_path / "nope.json")
    assert nul.blocked_ids == set() and nul.blocked_usernames == set()
    assert not nul.is_blocked(author_id="999", username="blocked_synth")


def test_cache_returns_identical_scores(trained: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    nul = Nullifier(trained["model_dir"], trained["blocklist"])
    texts = ["you are garbage, log off", "great thread, thanks", "you are garbage, log off"]
    first = nul.score(texts)
    assert first[0] == first[2]

    def boom(*_a, **_k):  # a cache hit must not touch the encoder again
        raise AssertionError("encoder called on cached text")

    monkeypatch.setattr(clf_mod, "encode", boom)
    assert nul.score(texts) == first
    assert nul.score(list(reversed(texts))) == list(reversed(first))


def test_threshold_resolution_order(trained: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    assert Nullifier(trained["model_dir"], trained["blocklist"], threshold=0.9).threshold == 0.9
    nul = Nullifier(trained["model_dir"], trained["blocklist"])
    assert (nul.threshold, nul.threshold_source) == (0.5, "config")
    monkeypatch.setenv("GVON_THRESHOLD", "0.7")
    nul = Nullifier(trained["model_dir"], trained["blocklist"])
    assert (nul.threshold, nul.threshold_source) == (0.7, "env")
    assert Nullifier(trained["model_dir"], trained["blocklist"], threshold=0.9).threshold == 0.9
    assert clf_mod.resolve_threshold(None, {}) == (0.7, "env")
    monkeypatch.setenv("GVON_THRESHOLD", "")
    assert clf_mod.resolve_threshold(None, {}) == (0.5, "default")


def _model_copy(trained: dict, tmp_path: Path, **overrides: object) -> Path:
    d = tmp_path / "model"
    shutil.copytree(trained["model_dir"], d)
    cfg = json.loads((d / "config.json").read_text())
    cfg.update(overrides)
    (d / "config.json").write_text(json.dumps(cfg))
    return d


def test_disabled_text_model_is_blocklist_only(trained: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    d = _model_copy(trained, tmp_path, text_model_enabled=False, threshold=1.01)
    nul = Nullifier(d, trained["blocklist"])
    hostile = "you absolute idiot, everyone hates you"
    assert nul.text_model_enabled is False
    assert not nul.should_nullify(hostile)
    assert nul.should_nullify(hostile, author_id="999")
    monkeypatch.setenv("GVON_THRESHOLD", "0.5")  # explicit override re-enables text scoring
    assert Nullifier(d, trained["blocklist"]).should_nullify(hostile)


def test_low_info_texts_abstain(trained: dict) -> None:
    nul = Nullifier(trained["model_dir"], trained["blocklist"])
    assert nul.score(["@synthetic_owner https://t.co/abc123", "@a @b", ""]) == [0.0, 0.0, 0.0]
    assert not nul.should_nullify("@synthetic_owner https://t.co/xyz")
    # leading mentions are stripped, so the reply target does not change the score
    assert nul.score(["@synthetic_a you are garbage, log off"]) == nul.score(["@synthetic_b you are garbage, log off"])


def test_watchlist_lowers_threshold(trained: dict, tmp_path: Path) -> None:
    bl = tmp_path / "blocklist.json"
    shutil.copy(trained["blocklist"], bl)
    (tmp_path / "watchlist.json").write_text(json.dumps({"accounts": [{"id": "555", "username": "Watched_Synth"}]}))
    nul = Nullifier(trained["model_dir"], bl, threshold=0.6)
    nul.score = lambda texts: [0.5 for _ in texts]  # type: ignore[method-assign]
    assert nul.watch_ids == {"555"} and nul.threshold_for(author_id="555") == pytest.approx(0.45)
    assert nul.should_nullify("some synthetic words", author_id="555")
    assert nul.should_nullify("other synthetic words", username="@watched_synth")
    assert not nul.should_nullify("some synthetic words", author_id="556")


def test_author_running_mean(trained: dict) -> None:
    nul = Nullifier(trained["model_dir"], trained["blocklist"], threshold=0.6)
    scores = {"first synthetic post": 0.9, "second synthetic post": 0.5, "third synthetic post": 0.5}
    nul.score = lambda texts: [scores[t] for t in texts]  # type: ignore[method-assign]
    assert nul.should_nullify("first synthetic post", author_id="77")  # on its own
    assert not nul.should_nullify("second synthetic post", author_id="77")  # n=2: too few
    assert not nul.should_nullify("second synthetic post", author_id="77")  # same text is not counted twice
    assert nul.should_nullify("third synthetic post", author_id="77")  # n=3, mean 0.633 >= 0.6
    assert not nul.should_nullify("third synthetic post", author_id="78")  # other author unaffected


def test_confidence_weight() -> None:
    assert train_mod.confidence_weight(0.5) == 0.5
    assert train_mod.confidence_weight(1.0) == 1.0
    assert train_mod.confidence_weight(0.0) == 1.0
    assert train_mod.confidence_weight(None) == 1.0
    assert train_mod.confidence_weight("bad") == 1.0


def test_build_dataset_join_targets_and_skips() -> None:
    labels = {"1": {"id": "1", "author_id": "a1", "label": "nullify", "nullify_score": 0.9, "reasons": ["Hostility"]},
              "2": {"id": "2", "author_id": "a2", "label": "good", "nullify_score": 0.1, "reasons": ["support"]},
              "3": {"id": "3", "author_id": "a2", "label": "neutral", "nullify_score": 0.4, "reasons": ["info"]},
              "4": {"id": "4", "author_id": "a3", "label": "nullify", "nullify_score": 0.8, "reasons": ["hostility"]},
              "5": {"id": "5", "author_id": "a4", "label": "nullify", "nullify_score": 0.9, "reasons": ["spam_bot"]},
              "6": {"id": "6", "author_id": "a5", "label": "good", "nullify_score": 0.0, "reasons": ["support"]},
              "7": {"id": "7", "author_id": "a6", "label": "nullify", "nullify_score": 0.7, "reasons": ["novel tag"]}}
    raw = {"1": "@owner you fraud", "2": "@owner @x great work", "3": "plain info https://t.co/x",
           "5": "buy my coin", "6": "@owner https://t.co/only", "7": "@owner whatever this is"}
    ds = train_mod.build_dataset(labels, raw)
    assert ds.texts == ["you fraud", "great work", "plain info", "whatever this is"]
    assert ds.y.tolist() == [1, 0, 0, 1]
    assert ds.authors == ["a1", "a2", "a2", "a6"]
    assert ds.counts == {"nullify": 2, "good": 1, "neutral": 1, "skipped_no_text": 1,
                         "skipped_identity_nullify": 1, "skipped_low_info": 1}
    assert ds.unlisted_content_tags == {"novel_tag": 1}
    # classes carry equal total weight; a2's two rows are down-weighted by 1/sqrt(2)
    assert ds.w[ds.y == 1].sum() == pytest.approx(ds.w[ds.y == 0].sum())
    assert ds.w[1] / ds.w[2] == pytest.approx(0.9 / 0.6)
    all_ds = train_mod.build_dataset(labels, raw, target="all")
    assert all_ds.y.sum() == 3 and "skipped_identity_nullify" not in all_ds.counts


def test_grouped_folds_never_split_an_author_or_a_duplicate_text() -> None:
    authors, texts, ys = [], [], []
    for a in range(12):
        for j in range(3):
            authors.append(f"author{a}")
            texts.append(f"synthetic text {a}-{j}")
            ys.append(1 if a % 3 == 0 else 0)
    # two different authors posting the same copy-paste text must land in one group
    texts[0] = texts[3] = "identical synthetic copy paste"
    groups = train_mod.make_groups(authors, texts)
    assert groups[0] == groups[3]
    folds = train_mod.grouped_folds(np.asarray(ys), groups, 5)
    assert folds is not None
    for tr, va in folds:
        assert not ({authors[i] for i in tr} & {authors[i] for i in va})
        assert not ({texts[i] for i in tr} & {texts[i] for i in va})


def test_select_threshold_needs_precision_and_support() -> None:
    y = np.asarray([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    p = np.asarray([0.1, 0.2, 0.3, 0.75, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95])
    assert train_mod.select_threshold(y, p, 0.8, min_tp=5) == pytest.approx(0.4)  # 6 TP / 7 flagged
    assert train_mod.select_threshold(y, p, 0.9, min_tp=3) == pytest.approx(0.8)  # 0.75 is a negative
    assert train_mod.select_threshold(y, p, 0.9, min_tp=4) is None
    assert train_mod.decide_threshold(None, None, None) == (train_mod.DISABLED_THRESHOLD, "none_met_precision_target", False)
    assert train_mod.decide_threshold(None, 0.6, 0.8)[1:] == ("env", True)
    assert train_mod.decide_threshold(0.4, 0.6, 0.8) == (0.4, "cli", True)


def test_content_text() -> None:
    assert clf_mod.content_text("@a @b_c  hello   there https://t.co/x") == "hello there"
    assert clf_mod.content_text("hi @a") == "hi @a"
    assert clf_mod.is_low_info("@a https://t.co/x ok") and not clf_mod.is_low_info("@a fine")


def test_device_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GVON_DEVICE", "cpu")
    assert clf_mod.pick_device() == "cpu"


# --------------------------------------------------------------------------- multi-source training


def _split_sources(root: Path) -> tuple[Path, Path]:
    """The synthetic rows split into an X half (tweets.jsonl) and a Telegram half (telegram_tweets.jsonl).
    Telegram rows get "chat:msg" ids and author ids that collide with X author ids on purpose."""
    raw, labels = synthetic_rows()
    x_raw, x_lab, tg_raw, tg_lab = [], [], [], []
    for i, (r, lab) in enumerate(zip(raw, labels)):
        if i % 2:
            tid = f"-100{i}:{i}"
            tg_raw.append({**r, "id": tid, "author_id": f"a{i - 1}", "platform": "telegram"})
            tg_lab.append({**lab, "id": tid, "author_id": f"a{i - 1}"})
        else:
            x_raw.append(r)
            x_lab.append(lab)
    labels_dir, raw_dir = root / "data/labels", root / "data/raw"
    write_jsonl(raw_dir / "tweets.jsonl", x_raw)
    write_jsonl(labels_dir / "tweets.jsonl", x_lab)
    write_jsonl(raw_dir / "telegram.jsonl", tg_raw)
    write_jsonl(labels_dir / "telegram_tweets.jsonl", tg_lab)
    return labels_dir, raw_dir


def test_resolve_sources_discovers_and_maps_by_prefix(tmp_path: Path) -> None:
    labels_dir, raw_dir = _split_sources(tmp_path)
    (labels_dir / "_cache.jsonl").write_text("")  # private files are never training input
    srcs = train_mod.resolve_sources(None, None, labels_dir, raw_dir)
    assert [(s, lp.name, rp.name) for s, lp, rp in srcs] == [("x", "tweets.jsonl", "tweets.jsonl"),
                                                             ("telegram", "telegram_tweets.jsonl", "telegram.jsonl")]
    one = train_mod.resolve_sources(labels_dir / "telegram_tweets.jsonl", None, labels_dir, raw_dir)
    assert [(s, rp.name) for s, _, rp in one] == [("telegram", "telegram.jsonl")]
    with pytest.raises(ValueError):
        train_mod.resolve_sources([labels_dir / "tweets.jsonl", labels_dir / "telegram_tweets.jsonl"],
                                  [raw_dir / "tweets.jsonl"], labels_dir, raw_dir)
    with pytest.raises(FileNotFoundError):
        train_mod.resolve_sources(None, None, tmp_path / "empty", raw_dir)
    (raw_dir / "telegram.jsonl").unlink()
    with pytest.raises(FileNotFoundError):
        train_mod.resolve_sources(None, None, labels_dir, raw_dir)


def test_load_sources_namespaces_non_x_authors(tmp_path: Path) -> None:
    labels_dir, raw_dir = _split_sources(tmp_path)
    labels, raw, n_rows = train_mod.load_sources(train_mod.resolve_sources(None, None, labels_dir, raw_dir))
    assert n_rows == {"x": 41, "telegram": 41}
    assert labels["synth0"]["author_id"] == "a0" and labels["synth0"]["_source"] == "x"
    assert labels["-1001:1"]["author_id"] == "telegram:a0" and labels["-1001:1"]["_source"] == "telegram"
    assert raw["-1001:1"] and raw["synth0"]
    ds = train_mod.build_dataset(labels, raw)
    assert "a0" in ds.authors and "telegram:a0" in ds.authors
    assert ds.groups[ds.authors.index("a0")] != ds.groups[ds.authors.index("telegram:a0")]
    assert set(ds.counts_by_source) == {"x", "telegram"}
    assert sum(ds.counts_by_source["x"].values()) + sum(ds.counts_by_source["telegram"].values()) == 82
    assert ds.sources.count("telegram") == sum(v for k, v in ds.counts_by_source["telegram"].items()
                                               if not k.startswith("skipped"))


def test_train_merges_every_labels_file_by_default(tmp_path: Path) -> None:
    labels_dir, raw_dir = _split_sources(tmp_path)
    out = tmp_path / "models/latest"
    rc = train_mod.main(["--labels-dir", str(labels_dir), "--raw-dir", str(raw_dir), "--out", str(out),
                         "--threshold", "0.5"])
    assert rc == 0
    cfg = json.loads((out / "config.json").read_text())
    assert cfg["n_train"] == 82
    assert cfg["label_counts"] == {"nullify": 36, "good": 36, "neutral": 10}
    by_src = cfg["label_counts_by_source"]
    assert set(by_src) == {"x", "telegram"}
    assert sum(by_src["x"].values()) == 41 and sum(by_src["telegram"].values()) == 41
    assert [(s["source"], s["n_label_rows"], s["n_train"]) for s in cfg["label_sources"]] == [("x", 41, 41),
                                                                                             ("telegram", 41, 41)]
    assert cfg["n_authors"] == 82  # colliding raw author ids stay distinct across platforms
    assert cfg["metrics"]["split"] == "grouped_by_author"
    # explicit pairs still work, one --raw per --labels
    out2 = tmp_path / "models/x_only"
    assert train_mod.main(["--labels", str(labels_dir / "tweets.jsonl"), "--raw", str(raw_dir / "tweets.jsonl"),
                           "--out", str(out2), "--threshold", "0.5"]) == 0
    assert json.loads((out2 / "config.json").read_text())["label_counts_by_source"].keys() == {"x"}


def test_blocklist_platform_scoping(trained: dict, tmp_path: Path) -> None:
    bl = tmp_path / "blocklist.json"
    bl.write_text(json.dumps({"accounts": [
        {"id": "999", "username": "Legacy_X", "nullify_score": 0.9, "summary": "s"},  # no platform key => x
        {"id": "777", "username": "tg_scammer", "nullify_score": 0.9, "summary": "s", "platform": "telegram"}]}))
    (tmp_path / "watchlist.json").write_text(json.dumps({"accounts": [
        {"id": "555", "username": "tg_watch", "nullify_score": 0.5, "summary": "s", "platform": "telegram"}]}))
    assert clf_mod.load_blocklist(bl) == ({"999", "777"}, {"legacy_x", "tg_scammer"})
    assert clf_mod.load_blocklist(bl, platform="x") == ({"999"}, {"legacy_x"})
    assert clf_mod.load_blocklist(bl, platform="telegram") == ({"777"}, {"tg_scammer"})
    nul = Nullifier(trained["model_dir"], bl, threshold=0.6)
    assert nul.is_blocked(author_id="777") and nul.is_blocked(username="legacy_x")  # None = any platform
    assert nul.is_blocked(author_id="777", platform="telegram")
    assert not nul.is_blocked(author_id="777", platform="x")
    assert not nul.is_blocked(username="@Legacy_X", platform="telegram")
    assert nul.is_blocked(username="@Legacy_X", platform="X")
    assert nul.is_watched(author_id="555") and nul.is_watched(author_id="555", platform="telegram")
    assert not nul.is_watched(author_id="555", platform="x")
    assert nul.threshold_for(author_id="555", platform="telegram") == pytest.approx(0.6 - nul.watch_delta)
    kind = "thank you, this was wonderful and helpful"
    assert nul.should_nullify(kind, author_id="777", platform="telegram")
    assert not nul.should_nullify(kind, author_id="777", platform="x")
    assert nul.should_nullify(kind, author_id="777")


# --------------------------------------------------------------------------- student v2: author context


def test_author_scalars_with_missing_and_full_authors() -> None:
    from datetime import datetime, timezone

    k = len(clf_mod.AUTHOR_SCALAR_NAMES)
    width = k + len(clf_mod.AUTHOR_MISSING_NAMES)
    none = clf_mod.author_scalars(None)
    assert none.shape == (width,)
    assert np.isnan(none[:6]).all() and none[6] == 0.0  # has_bio is known: no bio
    assert none[k:].tolist() == [1.0] * len(clf_mod.AUTHOR_MISSING_NAMES)
    now = datetime(2026, 1, 11, tzinfo=timezone.utc)
    full = clf_mod.author_scalars({"bio": "  synthetic   bio ", "followers": 99, "following": 9, "tweet_count": 0,
                                   "created_at": "2026-01-01T00:00:00.000Z", "verified": True}, now)
    assert full[:k] == pytest.approx([np.log1p(99), np.log1p(9), 0.0, np.log1p(10), np.log1p(9.9), 1.0, 1.0])
    assert full[k:].tolist() == [0.0] * len(clf_mod.AUTHOR_MISSING_NAMES)
    # legacy X format, junk values and a future creation date
    odd = clf_mod.author_scalars({"followers": "lots", "following": -3, "tweet_count": True,
                                  "created_at": "Mon Jan 12 00:00:00 +0000 2026", "verified": "yes"}, now)
    assert odd[k:].tolist() == [1.0, 1.0, 1.0, 0.0, 1.0] and odd[3] == 0.0  # age clamped at 0
    assert clf_mod.parse_created_at("Wed Oct 10 20:19:24 +0000 2018").year == 2018
    assert clf_mod.parse_created_at("not a date") is None and clf_mod.parse_created_at(None) is None
    assert clf_mod.author_bio({"bio": " a \n b "}) == "a b" and clf_mod.author_bio({"bio": None}) == ""
    # scaling: unknown -> 0, indicators untouched, never NaN, even with an all-unknown column
    raw = clf_mod.author_scalar_matrix([None, {"followers": 10, "verified": False}, {"followers": 1000}])
    block = clf_mod.scale_author_block(raw, clf_mod.fit_author_scaler(raw))
    assert block.shape == (3, width) and np.isfinite(block).all()
    assert block[0, 0] == 0.0 and block[0, k] == 1.0 and block[1, k] == 0.0
    assert clf_mod.author_signature(None) != clf_mod.author_signature({"bio": "x"})
    tg = clf_mod.author_from_telegram_user({"id": 1, "description": None, "verified": True})
    assert tg == {"bio": None, "followers": None, "following": None, "tweet_count": None, "created_at": None,
                  "verified": True}


def _users_rows(raw: list[dict], coverage: int = 2) -> list[dict]:
    """Synthetic users.jsonl rows for every `coverage`-th author (others stay unknown). Hostile authors look
    like throwaways (new, no followers, no bio); supportive ones are older with a bio."""
    rows = []
    for i, r in enumerate(raw):
        if i % coverage:
            continue
        hostile = any(r["text"].startswith(h) for h in HOSTILE_OPENERS)
        rows.append({"id": r["author_id"], "username": r["author_username"], "name": "synthetic",
                     "description": "" if hostile else f"synthetic bio about gardening {i}",
                     "created_at": "2025-12-20T00:00:00.000Z" if hostile else "2015-01-01T00:00:00.000Z",
                     "public_metrics": {"followers_count": 1 if hostile else 500 + i, "following_count": 300,
                                        "tweet_count": 5 if hostile else 4000},
                     "verified": False})
    return rows


@pytest.fixture(scope="module")
def trained_v2(tmp_path_factory: pytest.TempPathFactory) -> dict:
    root = tmp_path_factory.mktemp("gvon_v2")
    raw, labels = synthetic_rows()
    for r in raw:
        r["created_at"] = "2026-01-05T00:00:00.000Z"
    raw_dir, labels_dir = root / "data/raw", root / "data/labels"
    write_jsonl(raw_dir / "tweets.jsonl", raw)
    write_jsonl(labels_dir / "tweets.jsonl", labels)
    write_jsonl(raw_dir / "users.jsonl", _users_rows(raw))
    blocklist = root / "data/blocklist.json"
    blocklist.write_text(json.dumps({"accounts": []}))
    out = root / "models/latest"
    assert train_mod.main(["--labels-dir", str(labels_dir), "--raw-dir", str(raw_dir), "--out", str(out),
                           "--threshold", "0.5"]) == 0
    return {"root": root, "model_dir": out, "blocklist": blocklist, "raw_dir": raw_dir, "labels_dir": labels_dir}


def test_train_v2_records_author_inputs(trained_v2: dict) -> None:
    out = trained_v2["model_dir"]
    cfg = json.loads((out / "config.json").read_text())
    assert cfg["student_inputs"] == ["tweet_text", "author_bio", "author_stats"]
    assert cfg["student_inputs_requested"] == "auto"
    assert (out / clf_mod.AUTHOR_SCALER_FILE).exists()
    width = len(clf_mod.AUTHOR_SCALAR_NAMES) + len(clf_mod.AUTHOR_MISSING_NAMES)
    assert cfg["feature_dim"] == 2 * 384 + width
    af = cfg["author_features"]
    assert af["scalars"] == list(clf_mod.AUTHOR_SCALAR_NAMES)
    assert af["missing_indicators"] == list(clf_mod.AUTHOR_MISSING_NAMES)
    cov = af["coverage_by_source"]["x"]
    assert cov["rows"] == 82 and cov["with_author"] == 41 and 0 < cov["with_bio"] < cov["with_author"]
    assert cfg["metrics"]["split"] == "grouped_by_author" and cfg["probability_calibrated"] is True


def test_v2_scores_with_and_without_author_context(trained_v2: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    nul = Nullifier(trained_v2["model_dir"], trained_v2["blocklist"])
    assert nul.uses_author and nul.uses_author_bio and nul.author_scaler is not None
    text = "what a weird synthetic remark about the weather"
    throwaway = {"bio": "", "followers": 0, "following": 300, "tweet_count": 3,
                 "created_at": "2026-01-01T00:00:00Z", "verified": False}
    regular = {"bio": "synthetic bio about gardening", "followers": 900, "following": 300, "tweet_count": 5000,
               "created_at": "2014-01-01T00:00:00Z", "verified": True}
    partial = {"bio": None, "followers": "n/a", "created_at": "garbage"}
    out = nul.score([text, text, text, text], authors=[None, throwaway, regular, partial])
    assert all(np.isfinite(p) and 0.0 <= p <= 1.0 for p in out)
    assert out[1] != out[2]  # the author block reaches the head
    assert nul.score([text]) == [out[0]]  # no authors == every author unknown
    assert nul.score(["@a https://t.co/x"], authors=[throwaway]) == [0.0]  # low-info still abstains
    hostile = "you absolute idiot, everyone hates you"
    assert nul.should_nullify(hostile, author=throwaway)
    assert not nul.should_nullify("thank you, this was wonderful and helpful", author=regular)
    with pytest.raises(ValueError):
        nul.score([text], authors=[None, None])

    def boom(*_a, **_k):  # cached (text, author) pairs never re-encode
        raise AssertionError("encoder called on cached input")

    monkeypatch.setattr(clf_mod, "encode", boom)
    assert nul.score([text, text], authors=[throwaway, regular]) == out[1:3]


def test_text_only_flag_and_v1_ignores_authors(trained_v2: dict, trained: dict, tmp_path: Path) -> None:
    out = tmp_path / "model"
    shutil.copytree(trained_v2["model_dir"], out)  # v1 retrain into a v2 dir removes the stale scaler
    assert train_mod.main(["--labels-dir", str(trained_v2["labels_dir"]), "--raw-dir", str(trained_v2["raw_dir"]),
                           "--out", str(out), "--threshold", "0.5", "--student-inputs", "text"]) == 0
    cfg = json.loads((out / "config.json").read_text())
    assert cfg["student_inputs"] == ["tweet_text"] and cfg["author_features"] is None
    assert not (out / clf_mod.AUTHOR_SCALER_FILE).exists()
    v1 = Nullifier(trained["model_dir"], trained["blocklist"])
    assert not v1.uses_author
    t = "you are garbage, log off"
    assert v1.score([t], authors=[{"bio": "anything", "followers": 5}]) == v1.score([t])


def test_default_platform_scopes_blocklist(tmp_path, monkeypatch):
    """A Nullifier built with platform="x" ignores Telegram-only blocklist entries unless asked explicitly."""
    import json
    from gvon import classifier as c

    bl = tmp_path / "blocklist.json"
    bl.write_text(json.dumps({"accounts": [
        {"id": "1", "username": "synthetic_x_hater", "platform": "x"},
        {"id": "2", "username": "synthetic_tg_hater", "platform": "telegram"},
    ]}))
    n = c.Nullifier.__new__(c.Nullifier)
    n.default_platform = "x"
    n.blocklist_path = bl
    n.watchlist_path = tmp_path / "watchlist.json"
    n._load_lists()
    assert n.is_blocked(username="synthetic_x_hater")
    assert not n.is_blocked(username="synthetic_tg_hater")
    assert n.is_blocked(username="synthetic_tg_hater", platform="telegram")
    n.default_platform = None
    assert n.is_blocked(username="synthetic_tg_hater")
