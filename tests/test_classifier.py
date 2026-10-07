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
