"""Unit tests for gvon.ingest normalization. All payloads here are SYNTHETIC, hand-written fixtures."""
from __future__ import annotations

from datetime import datetime, timezone

from gvon.ingest import build_queries, classify_kind, compute_start_time, full_text, merge_pages

SYNTHETIC_USERS = [
    {"id": "100", "username": "synthetic_handle", "name": "Synthetic Handle", "verified": False,
     "public_metrics": {"followers_count": 1, "following_count": 2, "tweet_count": 3, "listed_count": 9}},
    {"id": "200", "username": "synthetic_replier", "name": "Synthetic Replier"},
]


def synthetic_tweet(tid: str, author: str = "200", refs: list[dict] | None = None, **extra) -> dict:
    t = {"id": tid, "text": f"synthetic text {tid}", "author_id": author,
         "created_at": f"2026-01-0{int(tid) % 9 + 1}T00:00:00.000Z", "conversation_id": tid,
         "lang": "en", "public_metrics": {"like_count": 0, "reply_count": 0, "retweet_count": 0, "quote_count": 0}}
    if refs is not None:
        t["referenced_tweets"] = refs
    t.update(extra)
    return t


def test_kind_precedence_reply_over_quote_over_mention():
    both = synthetic_tweet("1", refs=[{"type": "quoted", "id": "9"}, {"type": "replied_to", "id": "8"}])
    assert classify_kind(both, own=False) == "reply"
    assert classify_kind(synthetic_tweet("2", refs=[{"type": "quoted", "id": "9"}]), own=False) == "quote"
    assert classify_kind(synthetic_tweet("3"), own=False) == "mention"
    assert classify_kind(synthetic_tweet("4", refs=[{"type": "replied_to", "id": "8"}]), own=True) == "own"


def test_note_tweet_overrides_truncated_text():
    t = synthetic_tweet("5", text="truncated…", note_tweet={"text": "synthetic full long post"})
    assert full_text(t) == "synthetic full long post"
    assert full_text(synthetic_tweet("6")) == "synthetic text 6"


def test_merge_dedupes_across_queries_and_splits_context():
    own_tweet = synthetic_tweet("10", author="100")
    reply = synthetic_tweet("11", refs=[{"type": "replied_to", "id": "10"}], in_reply_to_user_id="100")
    unseen_parent = synthetic_tweet("12", author="100")
    page_mentions = {"data": [reply], "includes": {"users": SYNTHETIC_USERS, "tweets": [own_tweet, unseen_parent]}}
    page_replies = {"data": [reply], "includes": {"users": SYNTHETIC_USERS[1:], "tweets": [own_tweet]}}
    page_own = {"data": [own_tweet], "includes": {"users": SYNTHETIC_USERS[:1]}}

    tweets, users, context = merge_pages([(False, page_mentions), (False, page_replies), (True, page_own)])

    assert [t["id"] for t in tweets].count("11") == 1
    by_id = {t["id"]: t for t in tweets}
    assert by_id["11"]["kind"] == "reply"
    assert by_id["11"]["author_username"] == "synthetic_replier"
    assert by_id["11"]["referenced"] == [{"type": "replied_to", "id": "10"}]
    assert by_id["10"]["kind"] == "own"
    # context = included tweets not already in tweets.jsonl
    assert [c["id"] for c in context] == ["12"]
    assert context[0]["kind"] == "context"
    assert sorted(u["id"] for u in users) == ["100", "200"]
    u100 = next(u for u in users if u["id"] == "100")
    assert u100["public_metrics"] == {"followers_count": 1, "following_count": 2, "tweet_count": 3}
    assert next(u for u in users if u["id"] == "200")["verified"] is False


def test_own_wins_when_same_id_seen_as_mention():
    t = synthetic_tweet("20", author="100")
    tweets, _, _ = merge_pages([(False, {"data": [t]}), (True, {"data": [t]})])
    assert len(tweets) == 1 and tweets[0]["kind"] == "own"


def test_queries_and_start_time_clamp():
    qs = {q.name: q.query for q in build_queries("@someone")}
    assert qs["A_mentions"] == "@someone -from:someone -is:retweet"
    assert qs["C_quotes"] == "url:x.com/someone -from:someone -is:retweet"
    now = datetime(2026, 1, 10, tzinfo=timezone.utc)
    assert compute_start_time(7, now) == "2026-01-03T00:02:00Z"
    assert compute_start_time(1, now) == "2026-01-09T00:00:00Z"


# --------------------------------------------------------------------------- resume / window (synthetic state)

import json  # noqa: E402
from datetime import timedelta  # noqa: E402
from pathlib import Path  # noqa: E402

import pytest  # noqa: E402

from gvon import ingest  # noqa: E402

NOW = datetime(2026, 10, 7, 0, 26, 22, tzinfo=timezone.utc)


def _state_on_disk(out: Path, start_time: str, *, completed: dict[str, bool] | None = None) -> dict:
    specs = build_queries("synthetic_handle")
    state = ingest.load_state(out, specs, start_time, "synthetic_handle", fresh=True, now=NOW)
    state["start_time"] = start_time
    for name, done in (completed or {}).items():
        state["queries"][name]["completed"] = done
    state["queries"]["A_mentions"]["next_token"] = "SYNTHETIC_TOKEN"
    state["queries"]["A_mentions"]["pages"] = 1
    ingest.append_page(out, "A_mentions", {"data": [{**synthetic_tweet("31"), "created_at": "2026-10-05T00:00:00.000Z"},
                                                    {**synthetic_tweet("32"), "created_at": "2026-10-03T10:00:00.000Z"}]})
    ingest.save_state(out, state)
    return state


def test_resume_with_stale_start_time_rolls_window_and_keeps_pages(tmp_path: Path) -> None:
    _state_on_disk(tmp_path, "2026-09-30T00:12:29Z", completed={"C_quotes": True})
    specs = build_queries("synthetic_handle")
    state = ingest.load_state(tmp_path, specs, "ignored", "synthetic_handle", fresh=False, now=NOW)
    assert state["start_time"] == compute_start_time(7, NOW) and not ingest.is_stale(state["start_time"], NOW)
    a = state["queries"]["A_mentions"]
    assert a["next_token"] is None and a["completed"] is False
    assert a["end_time"] == "2026-10-03T10:00:01Z"  # oldest saved tweet + 1s: continue, do not re-bill
    assert state["queries"]["B_replies"]["end_time"] is None  # nothing fetched yet: plain restart
    assert state["queries"]["C_quotes"]["completed"] is True
    assert (tmp_path / "_pages" / "A_mentions.jsonl").exists(), "saved pages must survive a resume"


def test_roll_window_completes_query_whose_range_is_used_up() -> None:
    state = {"since_days": 7, "start_time": "2026-09-30T00:00:00Z",
             "queries": {"A": {"completed": False, "next_token": "t", "oldest_created_at": "2026-09-29T00:00:00.000Z"}}}
    ingest.roll_window(state, NOW)
    assert state["queries"]["A"]["completed"] is True


def test_search_params_end_time() -> None:
    q = build_queries("x")[0]
    p = ingest.search_params(q, "2026-10-01T00:00:00Z", None, "2026-10-02T00:00:00Z")
    assert p["end_time"] == "2026-10-02T00:00:00Z" and "next_token" not in p


class FakeClient:
    """Replays scripted responses; records the params of each call."""

    def __init__(self, script: list) -> None:
        self.script = script
        self.calls: list[dict] = []

    def get(self, url: str, params) -> dict:
        p = params() if callable(params) else params
        self.calls.append(p)
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def test_start_time_400_rolls_instead_of_dropping_optional_query(tmp_path: Path) -> None:
    specs = [q for q in build_queries("synthetic_handle") if q.name == "C_quotes"]
    state = ingest.load_state(tmp_path, specs, compute_start_time(7), "synthetic_handle", fresh=True)
    client = FakeClient([ingest.ApiError(400, '{"errors":[{"parameters":{"start_time":["x"]}}]}'),
                         {"data": [], "meta": {"result_count": 0}}])
    ingest.fetch_all(client, specs, state, tmp_path)
    q = state["queries"]["C_quotes"]
    assert q["completed"] is True and q["dropped"] is None and q["pages"] == 1
    assert state["window_rolls"] == 1 and len(client.calls) == 2


def test_client_rebuilds_params_after_rate_limit_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    class R:
        def __init__(self, code: int) -> None:
            self.status_code, self.headers, self.text = code, {"x-rate-limit-reset": "1"}, ""

        def json(self) -> dict:
            return {"data": []}

    responses = [R(429), R(200)]
    seen: list[dict] = []
    c = ingest.Client("synthetic-not-a-token")
    monkeypatch.setattr(c._s, "get", lambda url, params, timeout: (seen.append(params), responses.pop(0))[1])
    monkeypatch.setattr(ingest.time, "sleep", lambda s: None)
    n = iter(range(10))
    assert c.get("u", lambda: {"call": next(n)}) == {"data": []}
    assert seen == [{"call": 0}, {"call": 1}]


def test_recent_completed_run_is_reused(tmp_path: Path) -> None:
    specs = build_queries("synthetic_handle")
    state = ingest.load_state(tmp_path, specs, "2026-10-01T00:00:00Z", "synthetic_handle", fresh=True)
    for q in state["queries"].values():
        q["completed"] = True
    state["finished_at"] = "2026-10-07T00:00:00Z"
    ingest.save_state(tmp_path, state)
    assert ingest.recent_completed_state(tmp_path, "synthetic_handle", specs, 6, NOW) is not None
    assert ingest.recent_completed_state(tmp_path, "synthetic_handle", specs, 6, NOW + timedelta(hours=7)) is None
    assert ingest.recent_completed_state(tmp_path, "other_handle", specs, 6, NOW) is None
    state["queries"]["A_mentions"]["completed"] = False
    ingest.save_state(tmp_path, state)
    assert ingest.recent_completed_state(tmp_path, "synthetic_handle", specs, 6, NOW) is None


def test_default_paths_are_repo_anchored() -> None:
    from gvon import classifier, env, label, train

    assert ingest.DEFAULT_OUT == env.ROOT / "data" / "raw"
    assert label.DEFAULT_LABELS == env.ROOT / "data" / "labels" and label.DEFAULT_RAW == env.ROOT / "data" / "raw"
    assert train.DEFAULT_OUT == env.ROOT / "models" / "latest"
    assert classifier.DEFAULT_MODEL_DIR == env.ROOT / "models" / "latest"
    assert json.dumps(str(classifier.DEFAULT_BLOCKLIST)).endswith('data/blocklist.json"')
