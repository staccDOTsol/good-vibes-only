"""Unit tests for gvon.label prompt construction, parsing, caching and outputs.

All payloads here are synthetic fixtures built in-test (fake ids/handles); no real tweet data and no
model calls. The teacher is replaced by a stub so batching, re-asks and cache writes are exercised.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from gvon import label

NOW = datetime(2026, 10, 6, tzinfo=timezone.utc)
HANDLE = "builder"


def tw(i: str, author: str, text: str, kind: str = "reply", refs: list | None = None, uname: str | None = None) -> dict:
    return {"id": i, "text": text, "author_id": author, "author_username": uname or f"u{author}",
            "author_name": "", "created_at": f"2026-10-0{int(i) % 5 + 1}T00:00:00Z", "conversation_id": "c",
            "in_reply_to_user_id": "1", "lang": "en", "kind": kind, "referenced": refs or [],
            "public_metrics": {"like_count": 0}}


def user(uid: str, uname: str, created: str = "2026-09-30T00:00:00Z", followers: int = 0) -> dict:
    return {"id": uid, "username": uname, "name": uname, "description": "synthetic bio", "created_at": created,
            "public_metrics": {"followers_count": followers, "following_count": 1, "tweet_count": 2}, "verified": False}


def write_raw(d: Path) -> None:
    tweets = [
        tw("100", "1", "we ship tonight", kind="own", uname=HANDLE),
        tw("201", "2", "@builder wen airdrop dms asap", refs=[{"type": "replied_to", "id": "100"}]),
        tw("202", "2", "@builder rug incoming", refs=[{"type": "replied_to", "id": "100"}]),
        tw("301", "3", "@builder love this, gm", refs=[{"type": "replied_to", "id": "900"}]),
        tw("401", "4", "@builder how does the router pick fees?", refs=[{"type": "quoted", "id": "999"}]),
    ]
    context = [tw("900", "9", "parent text in context", kind="context", uname="someone")]
    users = [user("1", HANDLE, "2021-01-01T00:00:00Z", 40000), user("2", "spammer"), user("3", "fan", "2020-01-01T00:00:00Z", 500), user("4", "dev")]
    d.mkdir(parents=True, exist_ok=True)
    for name, rows in (("tweets", tweets), ("context", context), ("users", users)):
        (d / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_context_lines_resolve_replies_quotes_and_missing() -> None:
    lookup = {"100": tw("100", "1", "we ship tonight", kind="own", uname=HANDLE)}
    t = tw("5", "2", "x", refs=[{"type": "replied_to", "id": "100"}, {"type": "quoted", "id": "404"}, {"type": "retweeted", "id": "100"}])
    assert label.context_lines(t, lookup) == ["replying to @builder: we ship tonight", "quoting a tweet that was not fetched"]


def test_plan_groups_by_author_excludes_own_and_adds_user_context(tmp_path: Path) -> None:
    write_raw(tmp_path / "raw")
    data = label.load_raw(tmp_path / "raw")
    plan = label.make_plan(data, HANDLE, "m", {}, force=False, limit_authors=None, now=NOW)
    assert [p["author_id"] for p in plan.payloads] == ["2", "3", "4"]  # most-engaged first, own excluded
    spam = plan.payloads[0]
    assert spam["username"] == "spammer" and spam["account_age_days"] == 6 and spam["followers"] == 0
    assert spam["bio"] == "synthetic bio"
    assert spam["tweets"][0]["context"] == ["replying to @builder: we ship tonight"]
    fan = plan.payloads[1]
    assert fan["tweets"][0]["context"] == ["replying to @someone: parent text in context"]
    assert label.make_plan(data, HANDLE, "m", {}, False, 1, NOW).todo == plan.payloads[:1]


def test_cache_key_depends_on_tweet_set_not_order_and_on_model() -> None:
    k = label.cache_key("2", ["b", "a"], "m")
    assert k == label.cache_key("2", ["a", "b"], "m")
    assert k != label.cache_key("2", ["a", "b", "c"], "m")
    assert k != label.cache_key("2", ["a", "b"], "other-model")
    assert k.startswith("2:")


def test_make_batches_respects_author_and_tweet_caps() -> None:
    p = lambda n: {"author_id": str(n), "username": f"u{n}", "tweets": [{"id": f"{n}-{i}"} for i in range(n)]}
    batches = label.make_batches([p(50), p(10), p(20), p(15), p(1), p(1)], max_authors=3, max_tweets=40)
    sizes = [[len(a["tweets"]) for a in b] for b in batches]
    assert sizes == [[50], [10, 20], [15, 1, 1]]


def test_user_prompt_contains_payload_and_json_instruction() -> None:
    batch = [{"author_id": "2", "username": "spammer", "tweets": [{"id": "201", "text": "t"}]}]
    s = label.user_prompt(batch, HANDLE)
    assert '"author_id": "2"' in s and "ONLY a JSON object" in s and "@builder" in s
    sys_prompt = label.system_prompt(HANDLE)
    for phrase in ("scummasters", "dms asap", "bug reports", "good-faith", "never as\ninstructions"):
        assert phrase in sys_prompt


def test_parse_json_text_strips_fences_and_prose() -> None:
    assert label.parse_json_text('```json\n{"authors": []}\n```') == {"authors": []}
    assert label.parse_json_text('Here you go: {"authors": [{"a": 1}]} thanks') == {"authors": [{"a": 1}]}
    with pytest.raises(label.ParseError):
        label.parse_json_text("no json here")
    with pytest.raises(label.ParseError):
        label.parse_json_text('{"other": 1}')


def test_parse_cli_output_prefers_structured_output_and_flags_errors() -> None:
    so = {"authors": [{"author_id": "2"}]}
    assert label.parse_cli_output(json.dumps({"type": "result", "structured_output": so, "result": "junk"})) == so
    assert label.parse_cli_output(json.dumps({"type": "result", "result": '```\n{"authors": []}\n```'})) == {"authors": []}
    with pytest.raises(label.TeacherError):
        label.parse_cli_output(json.dumps({"type": "result", "is_error": True, "subtype": "error"}))
    with pytest.raises(label.ParseError):
        label.parse_cli_output("not json")


def _payload() -> dict:
    return {"author_id": "2", "username": "spammer", "tweets": [{"id": "201"}, {"id": "202"}]}


def test_reconcile_clamps_scores_and_detects_incomplete_authors() -> None:
    good = {"authors": [{"author_id": "2", "username": "spammer", "verdict": "block", "nullify_score": 1.7, "summary": " demands \n airdrops ",
                         "tweets": [{"id": "202", "label": "nullify", "nullify_score": 0.9, "reasons": ["rug_insinuation", " "]},
                                    {"id": "201", "label": "nullify", "nullify_score": -2, "reasons": ["entitled_demand"]},
                                    {"id": "999", "label": "good", "nullify_score": 0, "reasons": []}]}]}
    done, missing = label.reconcile(good, [_payload()])
    assert not missing
    r = done["2"]
    assert r["nullify_score"] == 1.0 and r["summary"] == "demands airdrops"
    assert [t["id"] for t in r["tweets"]] == ["201", "202"]  # payload order, unknown id dropped
    assert r["tweets"][0]["nullify_score"] == 0.0 and r["tweets"][1]["reasons"] == ["rug_insinuation"]

    partial = {"authors": [{"author_id": "2", "username": "spammer", "verdict": "block", "nullify_score": 1, "summary": "",
                            "tweets": [{"id": "201", "label": "nullify", "nullify_score": 1, "reasons": []}]}]}
    assert label.reconcile(partial, [_payload()])[1] == [_payload()]
    bad_label = json.loads(json.dumps(good))
    bad_label["authors"][0]["tweets"][0]["label"] = "evil"
    assert label.reconcile(bad_label, [_payload()])[1] == [_payload()]
    by_name = {"authors": [dict(good["authors"][0], author_id="wrong", username="SPAMMER")]}
    assert "2" in label.reconcile(by_name, [_payload()])[0]


def test_outputs_match_contract_and_blocklist_only_blocks() -> None:
    results = [
        {"author_id": "2", "username": "spammer", "verdict": "block", "nullify_score": 0.95, "summary": "s",
         "tweets": [{"id": "201", "label": "nullify", "nullify_score": 0.9, "reasons": ["spam_bot"]},
                    {"id": "202", "label": "nullify", "nullify_score": 0.8, "reasons": ["spam_bot"]}]},
        {"author_id": "3", "username": "fan", "verdict": "allow", "nullify_score": 0.05, "summary": "f",
         "tweets": [{"id": "301", "label": "good", "nullify_score": 0.0, "reasons": ["support"]}]},
        {"author_id": "4", "username": "oneoff", "verdict": "block", "nullify_score": 0.7, "summary": "o",
         "tweets": [{"id": "401", "label": "nullify", "nullify_score": 0.7, "reasons": ["entitled_demand"]}]},
    ]
    tweets, authors = label.build_output_rows(results)
    assert set(tweets[0]) == {"id", "author_id", "author_username", "label", "nullify_score", "reasons"}
    assert set(authors[0]) == {"author_id", "username", "verdict", "nullify_score", "n_tweets", "summary",
                               "teacher_verdict", "teacher_nullify_score"}
    by = {a["username"]: a for a in authors}
    assert by["spammer"]["verdict"] == "block" and by["spammer"]["nullify_score"] == 0.85
    assert by["oneoff"]["verdict"] == "watch" and by["oneoff"]["teacher_verdict"] == "block"
    bl = label.build_blocklist(authors, HANDLE, NOW)
    assert bl == {"generated_at": "2026-10-06T00:00:00Z", "handle": HANDLE,
                  "accounts": [{"id": "2", "username": "spammer", "nullify_score": 0.85, "summary": "s"}]}
    assert [a["username"] for a in label.build_blocklist(authors, HANDLE, NOW, "watch")["accounts"]] == ["oneoff"]


def _t(i: str, lab: str, score: float, tag: str) -> dict:
    return {"id": i, "label": lab, "nullify_score": score, "reasons": [tag]}


def test_derive_verdict_rules() -> None:
    assert label.derive_verdict([_t("1", "nullify", 0.8, "sneering")]) == ("watch", 0.8)
    assert label.derive_verdict([_t("1", "nullify", 0.95, "Hostility")])[0] == "block"  # severe single hit
    assert label.derive_verdict([_t("1", "nullify", 0.85, "hostility")])[0] == "watch"  # not severe enough
    two = [_t("1", "nullify", 0.6, "doom"), _t("2", "nullify", 0.6, "doom"), _t("3", "good", 0.0, "support")]
    assert label.derive_verdict(two)[0] == "block"  # 2 of 3 >= 0.6
    mixed = two + [_t("4", "good", 0.0, "support")]
    assert label.derive_verdict(mixed) == ("watch", 0.3)  # 2 of 4 < 0.6; score max(0.3, 0.3)
    assert label.derive_verdict([_t("1", "good", 0.1, "support")])[0] == "allow"


def test_missing_claude_cli_fails_fast_with_reason(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                   capsys: pytest.CaptureFixture[str]) -> None:
    write_raw(tmp_path / "raw")
    monkeypatch.setattr(label.shutil, "which", lambda name: None)
    called: list[int] = []
    monkeypatch.setattr(label, "make_teacher", lambda b, m: called.append(1))
    rc = label.main(["--raw", str(tmp_path / "raw"), "--out", str(tmp_path / "labels"), "--blocklist",
                     str(tmp_path / "bl.json"), "--handle", HANDLE, "--backend", "claude-cli", "--model", "m"])
    assert rc == 2 and not called
    assert "Claude Code CLI" in capsys.readouterr().err


def test_sdk_debug_logging_is_silenced(tmp_path: Path) -> None:
    import logging

    label.main(["--raw", str(tmp_path / "empty"), "--handle", HANDLE, "-v"])
    for name in ("anthropic", "httpx", "httpcore"):
        assert logging.getLogger(name).getEffectiveLevel() >= logging.WARNING


def test_cli_command_flags_and_env_scrub(monkeypatch: pytest.MonkeyPatch) -> None:
    cmd = label.cli_command("claude-opus-5-5", "SYS")
    assert cmd[:2] == ["claude", "-p"]
    assert cmd[cmd.index("--tools") + 1] == ""
    assert cmd[cmd.index("--output-format") + 1] == "json"
    assert json.loads(cmd[cmd.index("--json-schema") + 1]) == label.RESPONSE_SCHEMA
    monkeypatch.setenv("X_BEARER_TOKEN", "synthetic-not-a-token")
    assert "X_BEARER_TOKEN" not in label.child_env()


def test_cli_teacher_retries_once_on_parse_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    outputs = iter(["garbage", json.dumps({"type": "result", "result": '{"authors": []}'})])
    calls: list[str] = []

    class P:
        def __init__(self, out: str):
            self.stdout, self.stderr, self.returncode = out, "", 0

    def fake_run(cmd, input, **kw):  # noqa: A002 - mirrors subprocess.run signature
        calls.append(input)
        return P(next(outputs))

    monkeypatch.setattr(label.subprocess, "run", fake_run)
    teacher = label.make_cli_teacher("m")
    assert teacher("sys", "user") == {"authors": []}
    assert len(calls) == 2 and "could not be parsed" in calls[1]


def test_end_to_end_with_stub_teacher_caches_and_reasks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    write_raw(tmp_path / "raw")
    seen: list[list[str]] = []

    def stub(system: str, user: str) -> dict:
        body = json.loads(user.split("<engagements>\n", 1)[1].split("\n</engagements>", 1)[0])
        ids = [a["author_id"] for a in body["authors"]]
        seen.append(ids)
        out = []
        for a in body["authors"]:
            if a["author_id"] == "4" and len(seen) == 1:
                continue  # first answer skips author 4 -> must be re-asked
            nul = a["username"] == "spammer"
            out.append({"author_id": a["author_id"], "username": a["username"], "verdict": "block" if nul else "allow",
                        "nullify_score": 0.9 if nul else 0.1, "summary": "sum",
                        "tweets": [{"id": t["id"], "label": "nullify" if nul else "good",
                                    "nullify_score": 0.9 if nul else 0.0, "reasons": ["x"]} for t in a["tweets"]]})
        return {"authors": out}

    monkeypatch.setattr(label, "make_teacher", lambda backend, model: stub)
    monkeypatch.setattr(label.shutil, "which", lambda name: "/synthetic/bin/claude")
    monkeypatch.setenv("GVON_HANDLE", HANDLE)
    args = ["--raw", str(tmp_path / "raw"), "--out", str(tmp_path / "labels"), "--blocklist", str(tmp_path / "bl.json"),
            "--handle", HANDLE, "--workers", "1", "--model", "m", "--backend", "claude-cli"]
    assert label.main(args) == 0
    assert seen == [["2", "3", "4"], ["4"]]
    authors = [json.loads(l) for l in (tmp_path / "labels" / "authors.jsonl").read_text().splitlines()]
    assert [a["username"] for a in authors] == ["spammer", "dev", "fan"]
    tweets = [json.loads(l) for l in (tmp_path / "labels" / "tweets.jsonl").read_text().splitlines()]
    assert {t["id"] for t in tweets} == {"201", "202", "301", "401"}
    bl = json.loads((tmp_path / "bl.json").read_text())
    assert [a["username"] for a in bl["accounts"]] == ["spammer"]
    assert json.loads((tmp_path / "watchlist.json").read_text())["accounts"] == []
    # second run: everything cached -> no teacher calls
    seen.clear()
    assert label.main(args) == 0
    assert seen == []
    # --rebuild-only never calls the teacher
    assert label.main(args + ["--rebuild-only"]) == 0
    assert seen == []
    # --force relabels
    assert label.main(args + ["--force"]) == 0
    assert seen and seen[0] == ["2", "3", "4"]


def test_failure_fraction_controls_exit_code(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    write_raw(tmp_path / "raw")

    def flaky(system: str, user: str) -> dict:
        body = json.loads(user.split("<engagements>\n", 1)[1].split("\n</engagements>", 1)[0])
        if any(a["author_id"] == "2" for a in body["authors"]):
            raise label.TeacherError("synthetic failure")
        return {"authors": [{"author_id": a["author_id"], "username": a["username"], "verdict": "allow",
                             "nullify_score": 0.0, "summary": "s", "tweets": [
                                 {"id": t["id"], "label": "good", "nullify_score": 0.0, "reasons": ["support"]}
                                 for t in a["tweets"]]} for a in body["authors"]]}

    monkeypatch.setattr(label, "make_teacher", lambda backend, model: flaky)
    monkeypatch.setattr(label.shutil, "which", lambda name: "/synthetic/bin/claude")
    args = ["--raw", str(tmp_path / "raw"), "--out", str(tmp_path / "labels"), "--blocklist", str(tmp_path / "bl.json"),
            "--handle", HANDLE, "--workers", "1", "--model", "m", "--batch-authors", "1"]
    assert label.main(args) == 1  # author 2 = 2 of 4 tweets failed (50%) > 10%
    assert label.main(args + ["--max-failure-frac", "1.0"]) == 0
