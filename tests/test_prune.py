"""Tests for gvon.prune against SYNTHETIC fixtures (fixtures/*.json, fake @synthetic_* handles)."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from gvon.prune import BlocklistDecider, prune

FIX = Path(__file__).resolve().parent.parent / "fixtures"
HATERS = {"synthetic_hater_1", "synthetic_hater_2", "synthetic_hater_3"}
MARKER = "SYNTHETIC_TOXIC_MARKER"


class StubDecider:
    """Same interface as gvon.classifier.Nullifier (score + should_nullify), no is_blocked: nullifies by
    username set or a marker string in the text. Records calls so tests can check what was judged."""

    def __init__(self, usernames: set[str] = HATERS) -> None:
        self.usernames = {u.lower() for u in usernames}
        self.calls: list[tuple[str, str | None, str | None]] = []
        self.scored: list[str] = []

    def score(self, texts: list[str]) -> list[float]:
        self.scored.extend(texts)
        return [1.0 if MARKER in t else 0.0 for t in texts]

    def should_nullify(self, text: str, *, author_id: str | None = None, username: str | None = None) -> bool:
        self.calls.append((text, author_id, username))
        return (username or "").lower() in self.usernames or MARKER in text


class ExplodingDecider(StubDecider):
    def should_nullify(self, text: str, *, author_id: str | None = None, username: str | None = None) -> bool:
        raise RuntimeError("boom")


def load(name: str) -> Any:
    return json.loads((FIX / name).read_text())


def entry_ids(payload: Any) -> list[str]:
    """All entryIds (entries and module items) in document order."""
    out: list[str] = []

    def walk(n: Any) -> None:
        if isinstance(n, dict):
            if isinstance(n.get("entryId"), str):
                out.append(n["entryId"])
            for v in n.values():
                walk(v)
        elif isinstance(n, list):
            for v in n:
                walk(v)

    walk(payload)
    return out


# ---------------------------------------------------------------------------------- GraphQL timeline

HOME_REMOVED = {
    "tweet-1002",  # hater_1 plain tweet
    "tweet-1003",  # hater_2 inside TweetWithVisibilityResults, screen_name only at user.core
    "tweet-1004",  # friend quoting hater_1 -> nested nullified tweet takes the entry
    "tweet-1006",  # marker only in note_tweet text (legacy.full_text truncated before it)
    "conversationthread-2000-tweet-2002",  # one bad item out of 3
    "conversationthread-3000-tweet-3001",
    "conversationthread-3000-tweet-3002",
    "conversationthread-3000",  # all items gone -> module dropped
    "who-to-follow-4000-user-4001",  # user-only item judged by identity
}


def test_home_timeline_prunes_exactly_expected_entries() -> None:
    payload = load("home_timeline.json")
    original = copy.deepcopy(payload)
    new, report = prune(payload, StubDecider(), text_scope="all")

    before, after = set(entry_ids(payload)), set(entry_ids(new))
    assert before - after == HOME_REMOVED
    assert {r["entryId"] for r in report} == HOME_REMOVED
    assert len(report) == len(HOME_REMOVED)
    assert payload == original, "input must not be mutated"

    # cursors survive byte-for-byte
    cursors_before = [e for e in payload["data"]["home"]["home_timeline_urt"]["instructions"][1]["entries"]
                      if e["entryId"].startswith("cursor-")]
    cursors_after = [e for e in new["data"]["home"]["home_timeline_urt"]["instructions"][1]["entries"]
                     if e["entryId"].startswith("cursor-")]
    assert cursors_before == cursors_after and len(cursors_after) == 2

    # module with one bad item keeps the other two, in order
    entries = {e["entryId"]: e for e in new["data"]["home"]["home_timeline_urt"]["instructions"][1]["entries"]}
    assert [i["entryId"] for i in entries["conversationthread-2000"]["content"]["items"]] == [
        "conversationthread-2000-tweet-2001", "conversationthread-2000-tweet-2003"]
    assert [i["entryId"] for i in entries["who-to-follow-4000"]["content"]["items"]] == [
        "who-to-follow-4000-user-4002"]
    # non-entry instructions and metadata untouched
    assert new["data"]["home"]["home_timeline_urt"]["instructions"][0] == {"type": "TimelineClearCache"}
    assert new["data"]["home"]["home_timeline_urt"]["metadata"] == payload["data"]["home"]["home_timeline_urt"]["metadata"]


def test_report_rows_have_username_and_reason() -> None:
    _, report = prune(load("home_timeline.json"), StubDecider(), text_scope="all")
    rows = {r["entryId"]: r for r in report}
    assert rows["tweet-1002"]["username"] == "synthetic_hater_1"
    assert rows["tweet-1003"]["username"] == "synthetic_hater_2"
    assert rows["tweet-1004"]["username"] == "synthetic_hater_1"
    assert rows["tweet-1006"]["username"] == "synthetic_friend_3"
    assert rows["who-to-follow-4000-user-4001"]["username"] == "synthetic_hater_3"
    assert all(set(r) == {"entryId", "username", "reason"} and r["reason"] for r in report)


def test_note_tweet_text_is_what_gets_judged() -> None:
    stub = StubDecider(set())
    _, report = prune(load("home_timeline.json"), stub, text_scope="all")
    assert [r["entryId"] for r in report] == ["tweet-1006"]
    judged = {t for t, _, _ in stub.calls}
    assert any(MARKER in t for t in judged)
    assert "a long thread start and the full long body that is perfectly fine" in judged


def test_untouched_payload_is_returned_identically() -> None:
    for name in ("home_timeline.json", "tweet_detail.json", "notifications_all.json"):
        payload = load(name)
        raw = json.dumps(payload, sort_keys=True)
        new, report = prune(payload, _NoMarker())
        assert report == []
        assert new is payload
        assert json.dumps(new, sort_keys=True) == raw


class _NoMarker(StubDecider):
    """Nullifies nothing at all (the home fixture contains a marker text, so StubDecider(set()) would hit)."""

    def should_nullify(self, text: str, *, author_id: str | None = None, username: str | None = None) -> bool:
        return False


def test_owner_tweets_are_never_judged() -> None:
    stub = StubDecider()
    prune(load("home_timeline.json"), stub, owner_usernames=["synthetic_owner"])
    assert all(u != "synthetic_owner" for _, _, u in stub.calls)


def test_decision_cache_is_used_across_calls() -> None:
    cache: dict[str, bool] = {}
    stub = StubDecider()
    prune(load("home_timeline.json"), stub, cache=cache, text_scope="all")
    n_first = len(stub.calls)
    assert n_first > 0 and cache.get("tweet:1002") is True and cache.get("tweet:1001") is False
    _, report = prune(load("home_timeline.json"), stub, cache=cache, text_scope="all")
    assert len(stub.calls) == n_first, "second pass must be served from the cache"
    assert {r["entryId"] for r in report} == HOME_REMOVED


def test_score_is_called_once_as_batch_warmup() -> None:
    stub = StubDecider()
    prune(load("home_timeline.json"), stub, text_scope="all")
    assert stub.scored, "score() should be called to batch-warm the classifier cache"


# ---------------------------------------------------------------------------------- TweetDetail


def test_tweet_detail_owner_focal_tweet_is_protected() -> None:
    payload = load("tweet_detail.json")
    new, report = prune(payload, StubDecider(), owner_usernames=["Synthetic_Owner"])
    removed = set(entry_ids(payload)) - set(entry_ids(new))
    assert removed == {"conversationthread-5001-tweet-5011", "conversationthread-5002-tweet-5021",
                       "conversationthread-5002"}
    assert "tweet-5000" in entry_ids(new)
    assert "cursor-bottom-7990" in entry_ids(new)
    assert len(report) == 3
    assert new["data"]["threaded_conversation_with_injections_v2"]["instructions"][1] == {
        "type": "TimelineTerminateTimeline", "direction": "Top"}


def test_tweet_detail_owner_by_id_also_protects() -> None:
    new, _ = prune(load("tweet_detail.json"), StubDecider(), owner_ids=["900000000000000001"])
    assert "tweet-5000" in entry_ids(new)


def test_tweet_detail_without_owner_drops_focal_quoting_hater() -> None:
    new, report = prune(load("tweet_detail.json"), StubDecider())
    assert "tweet-5000" not in entry_ids(new)
    assert len(report) == 4


# ---------------------------------------------------------------------------------- globalObjects


def test_notifications_global_objects() -> None:
    payload = load("notifications_all.json")
    original = copy.deepcopy(payload)
    new, report = prune(payload, StubDecider(), owner_usernames=["synthetic_owner"])
    assert payload == original
    go = new["globalObjects"]
    assert set(go["tweets"]) == {"8001", "8003"}
    assert set(go["users"]) == {"900000000000000001", "900000000000000011"}
    # one aggregate policy (same as GraphQL): an aggregate naming a blocklisted user is dropped whole
    assert set(go["notifications"]) == {"n3"}

    entries = new["timeline"]["instructions"][1]["addEntries"]["entries"]
    ids = [e["entryId"] for e in entries]
    assert ids == ["cursor-top-1790000000009", "notification-n3",
                   "notification-t8003", "cursor-bottom-1790000000000"]
    assert new["timeline"]["instructions"][0] == {"clearCache": {}}
    assert new["timeline"]["instructions"][2] == payload["timeline"]["instructions"][2]

    assert len(report) == 3
    by_id = {r["entryId"]: r for r in report}
    assert by_id["notification-t8002"]["username"] == "synthetic_hater_1"
    assert by_id["notification-n2"]["username"] == "synthetic_hater_2"
    assert by_id["notification-n1"]["username"] == "synthetic_hater_1"


# ---------------------------------------------------------------------------------- totality


@pytest.mark.parametrize("payload", [
    {"data": {"home": {"home_timeline_urt": {"instructions": [{"type": "TimelineAddEntries", "entries": [
        {"entryId": 5, "content": None}, {"entryId": "tweet-1", "content": {"items": "not-a-list"}}, "junk", None]}]}}}},
    {"globalObjects": {"users": [], "tweets": "nope", "notifications": None}, "timeline": {"instructions": "x"}},
    {"globalObjects": {"tweets": {"1": {"full_text": "x", "user_id_str": "2"}}}, "timeline": {"instructions": [
        {"addEntries": {"entries": [{"entryId": "notification-1", "content": "not-a-dict"}]}}]}},
    [1, 2, {"entryId": "cursor-top-1"}],
    "just a string",
    None,
])
def test_malformed_payload_returns_original(payload: Any) -> None:
    snapshot = json.dumps(payload, sort_keys=True)
    new, report = prune(payload, StubDecider())
    assert json.dumps(new, sort_keys=True) == snapshot
    assert report == []


def test_decider_exception_returns_original_object() -> None:
    payload = load("home_timeline.json")
    new, report = prune(payload, ExplodingDecider())
    assert new is payload and report == []


def test_blocklist_decider(tmp_path: Path) -> None:
    bl = tmp_path / "blocklist.json"
    bl.write_text(json.dumps({"accounts": [{"id": "900000000000000022", "username": "SOMEONE_ELSE"},
                                           {"id": "1", "username": "@Synthetic_Hater_1"}]}))
    d = BlocklistDecider(bl)
    new, report = prune(load("home_timeline.json"), d)
    removed = set(entry_ids(load("home_timeline.json"))) - set(entry_ids(new))
    # hater_1 by username, hater_2 by id; hater_3 not listed; marker text ignored (no model)
    assert "tweet-1002" in removed and "tweet-1003" in removed and "tweet-1006" not in removed
    assert "who-to-follow-4000-user-4001" not in removed


# ---------------------------------------------------------------------------------- text scope


def _gql_tweet(tid: str, uid: str, name: str, text: str, **legacy: Any) -> dict:
    return {"__typename": "Tweet", "rest_id": tid,
            "core": {"user_results": {"result": {"__typename": "User", "rest_id": uid, "core": {"screen_name": name}}}},
            "legacy": {"full_text": text, "id_str": tid, "user_id_str": uid, **legacy}}


def _gql_page(*entries: dict) -> dict:
    return {"_gvon_fixture": "SYNTHETIC", "data": {"home": {"home_timeline_urt": {"instructions": [
        {"type": "TimelineAddEntries", "entries": list(entries)}]}}}}


def _tweet_entry(t: dict) -> dict:
    return {"entryId": f"tweet-{t['rest_id']}", "content": {"entryType": "TimelineTimelineItem", "itemContent": {
        "itemType": "TimelineTweet", "tweet_results": {"result": t}}}}


def test_default_scope_scores_only_engagements_with_owner() -> None:
    owner = {"owner_ids": ["900"], "owner_usernames": ["synthetic_owner"]}
    page = _gql_page(
        _tweet_entry(_gql_tweet("1", "11", "synthetic_friend_1", f"followed account post {MARKER}")),
        _tweet_entry(_gql_tweet("2", "12", "synthetic_friend_2", f"@synthetic_owner {MARKER}",
                                in_reply_to_user_id_str="900")),
        _tweet_entry(_gql_tweet("3", "13", "synthetic_friend_3", f"hey {MARKER}",
                                entities={"user_mentions": [{"id_str": "1", "screen_name": "Synthetic_Owner"}]})),
        _tweet_entry({**_gql_tweet("4", "14", "synthetic_friend_4", f"quoting {MARKER}"),
                      "quoted_status_result": {"result": _gql_tweet("5", "900", "synthetic_owner", "my post")}}),
        _tweet_entry(_gql_tweet("6", "21", "synthetic_hater_1", "harmless words, not engaging the owner")),
    )
    stub = StubDecider()
    new, report = prune(page, stub, **owner)
    assert {r["entryId"] for r in report} == {"tweet-2", "tweet-3", "tweet-4", "tweet-6"}
    assert "tweet-1" in entry_ids(new)  # out of scope: identity-only, friend is not blocklisted
    assert not any(MARKER in t and "followed account" in t for t, _, _ in stub.calls)
    assert not any("followed account" in t for t in stub.scored), "out-of-scope text must not be warmed"
    new_all, report_all = prune(page, StubDecider(), text_scope="all", **owner)
    assert "tweet-1" in {r["entryId"] for r in report_all}


def test_user_nested_in_tweet_is_not_a_user_only_item() -> None:
    t = _gql_tweet("1001", "11", "synthetic_friend_1", "this clip is great", extended_entities={"media": [
        {"additional_media_info": {"source_user": {"user_results": {"result": {
            "__typename": "User", "rest_id": "21", "core": {"screen_name": "synthetic_hater_1"}}}}}}]})
    page = _gql_page(_tweet_entry(t), {"entryId": "who-to-follow-1", "content": {"items": [
        {"entryId": "who-to-follow-1-user-21", "item": {"itemContent": {"itemType": "TimelineUser", "user_results": {
            "result": {"rest_id": "21", "core": {"screen_name": "synthetic_hater_1"}}}}}},
        {"entryId": "who-to-follow-1-user-11", "item": {"itemContent": {"itemType": "TimelineUser", "user_results": {
            "result": {"rest_id": "11", "core": {"screen_name": "synthetic_friend_1"}}}}}}]}})
    new, report = prune(page, StubDecider(), text_scope="all")
    assert [r["entryId"] for r in report] == ["who-to-follow-1-user-21"]
    assert "tweet-1001" in entry_ids(new)


def test_module_left_with_only_a_showmore_cursor_is_dropped() -> None:
    hater = _gql_tweet("801", "21", "synthetic_hater_1", "reply")
    module = {"entryId": "conversationthread-801", "content": {"entryType": "TimelineTimelineModule", "items": [
        {"entryId": "conversationthread-801-tweet-801", "item": {"itemContent": {
            "itemType": "TimelineTweet", "tweet_results": {"result": hater}}}},
        {"entryId": "conversationthread-801-cursor-showmore-123", "item": {"itemContent": {
            "itemType": "TimelineTimelineCursor", "cursorType": "ShowMore", "value": "SYNTHETIC_CURSOR"}}}]}}
    cursor_only = {"entryId": "conversationthread-900", "content": {"entryType": "TimelineTimelineModule", "items": [
        {"entryId": "conversationthread-900-cursor-showmore-1", "item": {"itemContent": {
            "itemType": "TimelineTimelineCursor", "value": "SYNTHETIC_CURSOR"}}}]}}
    new, report = prune(_gql_page(module, cursor_only), StubDecider())
    ids = entry_ids(new)
    assert "conversationthread-801" not in ids and "conversationthread-801-cursor-showmore-123" not in ids
    assert "conversationthread-900" in ids, "a module that never had content is left alone"
    assert "conversationthread-801" in {r["entryId"] for r in report}


# ---------------------------------------------------------------------------------- globalObjects integrity


def _rest_page(tweets: dict, users: dict, notifications: dict, entries: list) -> dict:
    return {"_gvon_fixture": "SYNTHETIC", "globalObjects": {"users": users, "tweets": tweets,
                                                            "notifications": notifications},
            "timeline": {"id": "SYNTHETIC", "instructions": [{"addEntries": {"entries": entries}}]}}


def _users(*pairs: tuple[str, str]) -> dict:
    return {uid: {"id_str": uid, "screen_name": name} for uid, name in pairs}


def _tweet_item(eid: str, tid: str) -> dict:
    return {"entryId": eid, "content": {"item": {"content": {"tweet": {"id": tid}}}}}


def _notif_item(eid: str, nid: str) -> dict:
    return {"entryId": eid, "content": {"item": {"content": {"notification": {"id": nid}}}}}


def _dangling(new: dict) -> list[str]:
    go = new["globalObjects"]
    out = []
    for tid, t in go["tweets"].items():
        for k in ("quoted_status_id_str", "retweeted_status_id_str"):
            if t.get(k) and t[k] not in go["tweets"]:
                out.append(f"{tid}.{k}")
        if t.get("user_id_str") not in go["users"]:
            out.append(f"{tid}.user")
    for nid, n in go.get("notifications", {}).items():
        agg = n.get("template", {}).get("aggregateUserActionsV1", {})
        for u in agg.get("fromUsers", []):
            if u["user"]["id"] not in go["users"]:
                out.append(f"{nid}.fromUsers")
        for o in agg.get("targetObjects", []):
            if o["tweet"]["id"] not in go["tweets"]:
                out.append(f"{nid}.target")
        for e in n.get("message", {}).get("entities", []):
            if e.get("ref", {}).get("user", {}).get("id") not in (None, *go["users"]):
                out.append(f"{nid}.entity")
    for ins in new["timeline"]["instructions"]:
        for e in ins.get("addEntries", {}).get("entries", []):
            tid = e.get("content", {}).get("item", {}).get("content", {}).get("tweet", {}).get("id")
            if tid and tid not in go["tweets"]:
                out.append(f"{e['entryId']}.tweet")
    return out


def test_global_objects_quote_chain_is_order_independent() -> None:
    users = _users(("33", "synthetic_friend_1"), ("55", "synthetic_friend_2"), ("44", "synthetic_friend_3"))
    tweets = {"700": {"user_id_str": "33", "full_text": "rt", "retweeted_status_id_str": "701"},
              "701": {"user_id_str": "55", "full_text": "look", "quoted_status_id_str": "702"},
              "702": {"user_id_str": "44", "full_text": MARKER}}
    page = _rest_page(tweets, users, {}, [_tweet_item("notification-700", "700")])
    new, report = prune(page, StubDecider(set()), text_scope="all")
    assert [r["entryId"] for r in report] == ["notification-700", "globalObjects.tweets.701",
                                              "globalObjects.tweets.702"]
    assert set(new["globalObjects"]["tweets"]) == set()
    # same verdicts whatever the dict order
    rev = _rest_page(dict(reversed(list(tweets.items()))), users, {}, [_tweet_item("notification-700", "700")])
    assert sorted(r["entryId"] for r in prune(rev, StubDecider(set()), text_scope="all")[1]) == sorted(
        r["entryId"] for r in report)
    assert _dangling(new) == []


def test_global_objects_never_dangles_owner_quote() -> None:
    users = _users(("1", "synthetic_owner"), ("33", "synthetic_friend_1"), ("21", "synthetic_hater_1"))
    tweets = {"500": {"user_id_str": "1", "full_text": "my take", "quoted_status_id_str": "600"},
              "600": {"user_id_str": "21", "full_text": "bad"},
              "601": {"user_id_str": "21", "full_text": "@synthetic_owner bad reply"}}
    notifs = {"n1": {"id": "n1", "message": {"text": "synthetic_friend_1 liked your post", "entities": []},
                     "template": {"aggregateUserActionsV1": {"targetObjects": [{"tweet": {"id": "500"}}],
                                                             "fromUsers": [{"user": {"id": "33"}}]}}}}
    page = _rest_page(tweets, users, notifs, [_notif_item("notification-n1", "n1"),
                                              _tweet_item("notification-t601", "601")])
    new, report = prune(page, StubDecider(), owner_usernames=["synthetic_owner"])
    assert [r["entryId"] for r in report] == ["notification-t601"]
    go = new["globalObjects"]
    assert "600" in go["tweets"] and "21" in go["users"], "the owner's quote must not dangle"
    assert "601" not in go["tweets"]
    assert "notification-n1" in entry_ids(new)
    assert _dangling(new) == []


def test_global_objects_aggregate_naming_blocked_user_is_dropped_whole() -> None:
    users = _users(("1", "synthetic_owner"), ("22", "synthetic_hater_1"), ("33", "synthetic_friend_1"))
    tweets = {"500": {"user_id_str": "1", "full_text": "my post"}}
    notifs = {"n1": {"id": "n1", "message": {"text": "synthetic_hater_1 and synthetic_friend_1 liked your post",
                                             "entities": [{"fromIndex": 0, "toIndex": 17, "ref": {"user": {"id": "22"}}},
                                                          {"fromIndex": 22, "toIndex": 40, "ref": {"user": {"id": "33"}}}]},
                     "template": {"aggregateUserActionsV1": {"targetObjects": [{"tweet": {"id": "500"}}],
                                                             "fromUsers": [{"user": {"id": "22"}}, {"user": {"id": "33"}}]}}},
              "n2": {"id": "n2", "message": {"text": "synthetic_friend_1 mentioned synthetic_hater_1", "entities": [
                  {"fromIndex": 30, "toIndex": 47, "ref": {"user": {"id": "22"}}}]},
                     "template": {"aggregateUserActionsV1": {"targetObjects": [], "fromUsers": [{"user": {"id": "33"}}]}}}}
    page = _rest_page(tweets, users, notifs, [_notif_item("notification-n1", "n1"), _notif_item("notification-n2", "n2")])
    new, report = prune(page, StubDecider(), owner_usernames=["synthetic_owner"])
    assert {r["entryId"] for r in report} == {"notification-n1", "notification-n2"}
    assert new["globalObjects"]["notifications"] == {}
    assert "22" not in new["globalObjects"]["users"]
    assert "synthetic_hater_1" not in json.dumps(new)
    assert _dangling(new) == []


def test_global_objects_unreferenced_nullified_tweet_still_counts_as_change() -> None:
    users = _users(("21", "synthetic_hater_1"))
    page = _rest_page({"9": {"user_id_str": "21", "full_text": "x"}}, users, {}, [])
    new, report = prune(page, StubDecider())
    assert new is not page and new["globalObjects"]["tweets"] == {}
    assert [r["entryId"] for r in report] == ["globalObjects.tweets.9"]
