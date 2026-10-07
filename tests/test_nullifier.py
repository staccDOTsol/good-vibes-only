"""Addon-level tests: real mitmproxy flow objects, SYNTHETIC fixture bodies, stub decider (no model)."""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest
from mitmproxy.test import tflow, tutils

from gvon import nullifier as nmod
from gvon.nullifier import GvonNullifier, owner_id_from_twid, sanitize_path, should_process

FIX = Path(__file__).resolve().parent.parent / "fixtures"
HATERS = {"synthetic_hater_1", "synthetic_hater_2", "synthetic_hater_3"}


class Stub:
    def score(self, texts: list[str]) -> list[float]:
        return [0.0] * len(texts)

    def should_nullify(self, text: str, *, author_id: str | None = None, username: str | None = None) -> bool:
        return (username or "").lower() in HATERS


def make_flow(body: bytes, *, host: str = "x.com", path: str = "/i/api/graphql/abc123/HomeTimeline?variables=%7B%7D",
              ctype: str = "application/json; charset=utf-8", status: int = 200, encoding: str | None = None,
              cookie: str | None = "twid=u%3D900000000000000001; ct0=fake"):
    req = tutils.treq(host=host, port=443, path=path.encode(), scheme=b"https")
    if cookie:
        req.headers["cookie"] = cookie
    resp = tutils.tresp(status_code=status, content=b"")
    resp.headers["content-type"] = ctype
    if encoding:
        resp.headers["content-encoding"] = encoding
    resp.content = body  # mitmproxy encodes per content-encoding
    return tflow.tflow(req=req, resp=resp)


def addon(monkeypatch: pytest.MonkeyPatch, **env: str) -> GvonNullifier:
    for k in ("GVON_RECORD", "GVON_DRY_RUN", "GVON_THRESHOLD", "GVON_HANDLE", "GVON_TEXT_SCOPE"):
        monkeypatch.setenv(k, env.get(k, ""))
    return GvonNullifier(decider=Stub())


def test_should_process() -> None:
    assert should_process("x.com", "/i/api/graphql/h/HomeTimeline?x=1", 200, "application/json")
    assert should_process("api.twitter.com", "/graphql/h/TweetDetail", 200, "application/json;charset=utf-8")
    assert should_process("twitter.com", "/i/api/2/notifications/all.json", 200, "application/json")
    assert not should_process("x.com", "/home", 200, "text/html")
    assert not should_process("abs.twimg.com", "/i/api/graphql/h/X", 200, "application/json")
    assert not should_process("x.com", "/i/api/graphql/h/X", 304, "application/json")
    assert not should_process("x.com", "/i/api/1.1/jot/client_event.json", 200, "application/json")


def test_twid_and_sanitize() -> None:
    assert owner_id_from_twid("u%3D900000000000000001") == "900000000000000001"
    assert owner_id_from_twid('"u=42"') == "42"
    assert owner_id_from_twid("garbage") is None and owner_id_from_twid(None) is None
    assert sanitize_path("/i/api/graphql/abc/HomeTimeline?variables=x") == "i_api_graphql_abc_HomeTimeline"


def test_response_is_pruned_and_reencoded(monkeypatch: pytest.MonkeyPatch) -> None:
    body = (FIX / "home_timeline.json").read_bytes()
    flow = make_flow(body, encoding="gzip")
    a = addon(monkeypatch)
    a.response(flow)
    assert flow.response.headers["content-encoding"] == "gzip"
    assert gzip.decompress(flow.response.raw_content)  # still valid gzip on the wire
    text = flow.response.get_text()
    assert "synthetic_hater_1" not in text and "synthetic_hater_3" not in text
    assert "cursor-bottom-1980" in text and "synthetic_friend_1" in text
    assert int(flow.response.headers["content-length"]) == len(flow.response.raw_content)


def test_owner_from_twid_cookie_protects_focal_tweet(monkeypatch: pytest.MonkeyPatch) -> None:
    flow = make_flow((FIX / "tweet_detail.json").read_bytes(), path="/i/api/graphql/h/TweetDetail")
    addon(monkeypatch).response(flow)
    data = json.loads(flow.response.get_text())
    ids = [e["entryId"] for e in data["data"]["threaded_conversation_with_injections_v2"]["instructions"][0]["entries"]]
    assert "tweet-5000" in ids and "conversationthread-5002" not in ids


def test_dry_run_leaves_body(monkeypatch: pytest.MonkeyPatch) -> None:
    body = (FIX / "home_timeline.json").read_bytes()
    flow = make_flow(body)
    addon(monkeypatch, GVON_DRY_RUN="1").response(flow)
    assert flow.response.content == body


def test_non_matching_and_bad_json_untouched(monkeypatch: pytest.MonkeyPatch) -> None:
    a = addon(monkeypatch)
    for flow in (make_flow(b"{not json"), make_flow((FIX / "home_timeline.json").read_bytes(), host="example.com"),
                 make_flow((FIX / "home_timeline.json").read_bytes(), ctype="text/html")):
        before = flow.response.content
        a.response(flow)
        assert flow.response.content == before


def test_record_writes_body_without_headers(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(nmod, "RECORD_DIR", tmp_path / "recordings")
    flow = make_flow((FIX / "notifications_all.json").read_bytes(), path="/i/api/2/notifications/all.json?count=20")
    addon(monkeypatch, GVON_RECORD="1").response(flow)
    files = list((tmp_path / "recordings").glob("*.json"))
    assert len(files) == 1 and files[0].name.endswith("_i_api_2_notifications_all.json.json")
    rec = json.loads(files[0].read_text())
    assert rec["request"] == {"method": "GET", "host": "x.com", "path": "/i/api/2/notifications/all.json"}
    assert "ct0" not in files[0].read_text() and "twid" not in files[0].read_text()
    assert rec["response"]["_gvon_fixture"].startswith("SYNTHETIC")


def _entry(tid: str, name: str, text_json: str) -> str:
    """One synthetic TimelineTweet entry as raw JSON text (so escapes like \\ud83d stay literal)."""
    return ('{"entryId":"tweet-%s","content":{"itemContent":{"tweet_results":{"result":{"rest_id":"%s",'
            '"core":{"user_results":{"result":{"rest_id":"u%s","core":{"screen_name":"%s"}}}},'
            '"legacy":{"full_text":%s}}}}}}' % (tid, tid, tid, name, text_json))


def test_lone_surrogate_body_is_still_filtered(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """X can emit an unpaired UTF-16 surrogate at a truncation boundary; the page must still be filtered."""
    body = ('{"_gvon_fixture":"SYNTHETIC","data":{"home":{"home_timeline_urt":{"instructions":[{"type":'
            '"TimelineAddEntries","entries":[' + _entry("1", "synthetic_friend_1", '"truncated \\ud83d"') + ","
            + _entry("2", "synthetic_hater_1", '"bad"') + ']}]}}}}').encode()
    monkeypatch.setattr(nmod, "RECORD_DIR", tmp_path / "recordings")
    flow = make_flow(body)
    addon(monkeypatch, GVON_RECORD="1").response(flow)
    out = json.loads(flow.response.get_text())
    ids = [e["entryId"] for e in out["data"]["home"]["home_timeline_urt"]["instructions"][0]["entries"]]
    assert ids == ["tweet-1"]
    assert out["data"]["home"]["home_timeline_urt"]["instructions"][0]["entries"][0]["content"]["itemContent"][
        "tweet_results"]["result"]["legacy"]["full_text"] == "truncated \ud83d"
    assert b"\\ud83d" in flow.response.content
    assert len(list((tmp_path / "recordings").glob("*.json"))) == 1


def test_text_scope_env(monkeypatch: pytest.MonkeyPatch) -> None:
    assert addon(monkeypatch).text_scope == "engagements"
    monkeypatch.setenv("GVON_TEXT_SCOPE", "all")
    assert GvonNullifier(decider=Stub()).text_scope == "all"
    monkeypatch.setenv("GVON_TEXT_SCOPE", "bogus")
    assert GvonNullifier(decider=Stub()).text_scope == "engagements"
