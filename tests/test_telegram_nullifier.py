"""Telegram sink tests: stub nullifier, fake client recording calls. No Telethon network, no real messages."""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any

import pytest
from telethon.errors import FloodWaitError
from telethon.tl.functions.account import UpdateNotifySettingsRequest
from telethon.tl.types import InputPeerUser

from gvon import telegram_nullifier as tn
from gvon.telegram_nullifier import TgConfig, TelegramSink, apply, decide, vault_tail

BAD = "synthetic hostile text for testing"
GOOD = "synthetic friendly text for testing"
BLOCKED_ID = 111
BLOCKED_NAME = "synthetic_blocked"


class StubNullifier:
    """Mirrors gvon.classifier.Nullifier's public API (no platform= parameter)."""
    threshold = 0.5

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def score(self, texts: list[str]) -> list[float]:
        return [0.9 if t == BAD else 0.1 for t in texts]

    def is_blocked(self, *, author_id: str | None = None, username: str | None = None) -> bool:
        return author_id == str(BLOCKED_ID) or (username or "").lower() == BLOCKED_NAME

    def should_nullify(self, text: str, *, author_id: str | None = None, username: str | None = None) -> bool:
        self.calls.append({"author_id": author_id, "username": username})
        return self.is_blocked(author_id=author_id, username=username) or self.score([text])[0] >= self.threshold


class PlatformStub(StubNullifier):
    def should_nullify(self, text: str, *, author_id: str | None = None, username: str | None = None,
                       platform: str | None = None) -> bool:
        self.calls.append({"author_id": author_id, "username": username, "platform": platform})
        return self.is_blocked(author_id=author_id, username=username) or self.score([text])[0] >= self.threshold


class FakeClient:
    def __init__(self, flood_once: bool = False) -> None:
        self.calls: list[tuple] = []
        self.flood_once = flood_once

    async def delete_messages(self, entity: Any, ids: Any, *, revoke: bool = True) -> None:
        if self.flood_once:
            self.flood_once = False
            raise FloodWaitError(request=None, capture=7)
        self.calls.append(("delete_messages", entity, list(ids), revoke))

    async def send_read_acknowledge(self, entity: Any, *, max_id: int | None = None) -> None:
        self.calls.append(("send_read_acknowledge", entity, max_id))

    async def get_input_entity(self, entity: Any) -> Any:
        self.calls.append(("get_input_entity", entity))
        return InputPeerUser(user_id=42, access_hash=0)

    async def edit_folder(self, entity: Any, folder: int) -> None:
        self.calls.append(("edit_folder", entity, folder))

    async def get_permissions(self, entity: Any, user: Any) -> Any:
        self.calls.append(("get_permissions", entity, user))
        return SimpleNamespace(is_admin=True, delete_messages=True)

    async def __call__(self, request: Any) -> None:
        self.calls.append(("request", request))

    async def download_media(self, message: Any, file: Any = None) -> str:
        self.calls.append(("download_media", getattr(message, "id", None)))
        from pathlib import Path

        out = Path(file) / "synthetic.jpg"
        out.write_bytes(b"synthetic-bytes")
        return str(out)

    def names(self) -> list[str]:
        return [c[0] for c in self.calls]


def msg(text: str, *, mid: int = 10, sender_id: int = 222) -> SimpleNamespace:
    return SimpleNamespace(id=mid, message=text, sender_id=sender_id, out=False, date=None)


DM = SimpleNamespace(id=5001, first_name="Synthetic", last_name="Person")
BASIC_GROUP = SimpleNamespace(id=6001, title="Synthetic basic group")
SUPERGROUP = SimpleNamespace(id=7001, title="Synthetic supergroup", megagroup=True, broadcast=False)
SENDER = SimpleNamespace(username="synthetic_sender")


def run(coro: Any) -> Any:
    return asyncio.run(coro)


async def no_sleep(_: float) -> None:
    return None


def sink(client: FakeClient, tmp_path, **cfg: Any) -> TelegramSink:
    return TelegramSink(client, StubNullifier(), TgConfig(**cfg), vault_path=tmp_path / "vault.jsonl", sleep=no_sleep)


# ------------------------------------------------------------------------------------------------- decide

def test_decide_defaults_and_blocklist() -> None:
    cfg = TgConfig()
    nul = StubNullifier()
    assert decide(GOOD, 222, "someone", True, False, nul, cfg) == (False, None, None)  # kept: never scored
    assert decide(BAD, 222, "someone", True, False, nul, cfg) == (True, 0.9, "delete_for_me")
    assert decide(BAD, 222, "someone", False, False, nul, cfg) == (True, 0.9, "log")
    # blocklist hit by id or username regardless of text
    assert decide(GOOD, BLOCKED_ID, None, True, False, nul, cfg)[0] is True
    assert decide(GOOD, 999, "Synthetic_Blocked", False, False, nul, cfg)[::2] == (True, "log")
    # admin + GVON_TG_GROUP_DELETE => delete for everyone; admin alone => group action
    assert decide(BAD, 222, None, False, True, nul, TgConfig(group_delete=True))[2] == "delete_for_everyone"
    assert decide(BAD, 222, None, False, True, nul, cfg)[2] == "log"
    assert decide(BAD, 222, None, True, True, nul, TgConfig(group_delete=True))[2] == "delete_for_me"


def test_platform_passed_only_when_accepted() -> None:
    plain, plat = StubNullifier(), PlatformStub()
    decide(BAD, 222, "u", True, False, plain, TgConfig())
    decide(BAD, 222, "u", True, False, plat, TgConfig())
    assert "platform" not in plain.calls[-1] and plain.calls[-1]["author_id"] == "222"
    assert plat.calls[-1]["platform"] == "telegram"


def test_config_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for k in ("GVON_TG_DM_ACTION", "GVON_TG_GROUP_ACTION", "GVON_TG_GROUP_DELETE", "GVON_DRY_RUN",
              "GVON_TG_MARK_READ", "GVON_TG_ALLOW", "GVON_TG_MEDIA_MAX_MB"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(tn.env, "load_env", lambda *a, **k: None)
    assert TgConfig.from_env() == TgConfig("delete_for_me", "log", False, False)
    monkeypatch.setenv("GVON_TG_DM_ACTION", "Mute")
    monkeypatch.setenv("GVON_TG_GROUP_ACTION", "archive")
    monkeypatch.setenv("GVON_TG_GROUP_DELETE", "1")
    monkeypatch.setenv("GVON_DRY_RUN", "1")
    assert TgConfig.from_env() == TgConfig("mute", "archive", True, True)
    monkeypatch.setenv("GVON_TG_MARK_READ", "1")
    monkeypatch.setenv("GVON_TG_ALLOW", "123, @Synthetic_Friend")
    monkeypatch.setenv("GVON_TG_MEDIA_MAX_MB", "5")
    cfg = TgConfig.from_env()
    assert cfg.mark_read is True and cfg.allow == frozenset({"123", "synthetic_friend"}) and cfg.media_max_mb == 5.0
    monkeypatch.setenv("GVON_TG_DM_ACTION", "nuke")
    with pytest.raises(ValueError):
        TgConfig.from_env()


# -------------------------------------------------------------------------------------------------- apply

def test_apply_delete_for_me_dm() -> None:
    c = FakeClient()
    assert run(apply("delete_for_me", c, DM, 10, sleep=no_sleep)) == "deleted_for_me"
    assert c.calls == [("delete_messages", DM, [10], False)]  # no read receipt, nothing else marked read


def test_apply_delete_for_everyone() -> None:
    c = FakeClient()
    assert run(apply("delete_for_everyone", c, BASIC_GROUP, [3, 4], sleep=no_sleep)) == "deleted_for_everyone"
    assert c.calls == [("delete_messages", BASIC_GROUP, [3, 4], True)]


def test_apply_mute_and_archive_once_per_chat() -> None:
    c, state = FakeClient(), {}
    run(apply("archive", c, BASIC_GROUP, 1, state=state, sleep=no_sleep))
    run(apply("archive", c, BASIC_GROUP, 2, state=state, sleep=no_sleep))
    assert c.names() == ["get_input_entity", "request", "edit_folder"]
    req = c.calls[1][1]
    assert isinstance(req, UpdateNotifySettingsRequest) and req.settings.mute_until == tn.MUTE_FOREVER
    assert c.calls[2] == ("edit_folder", BASIC_GROUP, 1)


def test_apply_delete_for_me_in_supergroup_falls_back_to_mute() -> None:
    c = FakeClient()
    res = run(apply("delete_for_me", c, SUPERGROUP, 9, is_channel=True, sleep=no_sleep))
    assert res.startswith("fallback_mute")
    assert "delete_messages" not in c.names() and "request" in c.names()


def test_apply_dry_run_and_log_make_no_calls() -> None:
    c = FakeClient()
    for action in ("delete_for_me", "mute", "archive", "log", "delete_for_everyone"):
        assert run(apply(action, c, DM, 1, dry_run=True)) == "dry_run"
    assert run(apply("log", c, DM, 1)) == "logged"
    assert c.calls == []


def test_apply_flood_wait_sleeps_and_retries() -> None:
    c, slept = FakeClient(flood_once=True), []

    async def rec_sleep(s: float) -> None:
        slept.append(s)

    assert run(apply("delete_for_me", c, DM, 10, sleep=rec_sleep)) == "deleted_for_me"
    assert slept == [8] and ("delete_messages", DM, [10], False) in c.calls


# --------------------------------------------------------------------------------------------- sink+vault

def test_sink_dm_default_deletes_for_me_and_vaults(tmp_path) -> None:
    c = FakeClient()
    s = sink(c, tmp_path)
    assert run(s.handle_message(msg(GOOD), DM, SENDER, is_private=True)) is None
    e = run(s.handle_message(msg(BAD), DM, SENDER, is_private=True))
    assert e and e["action"] == "delete_for_me"
    assert ("delete_messages", DM, [10], False) in c.calls
    v = vault_tail(10, tmp_path / "vault.jsonl")
    assert len(v) == 1 and v[0]["text"] == BAD and v[0]["result"] == "deleted_for_me"
    assert v[0]["sender_username"] == "synthetic_sender" and v[0]["chat_id"] == DM.id and v[0]["score"] == 0.9


def test_sink_group_default_only_logs(tmp_path) -> None:
    c = FakeClient()
    s = sink(c, tmp_path)
    e = run(s.handle_message(msg(BAD), BASIC_GROUP, SENDER, is_private=False))
    assert e["action"] == "log" and e["result"] == "logged"
    assert c.calls == []  # no admin lookup, no delete, no mute
    v = vault_tail(10, tmp_path / "vault.jsonl")
    assert len(v) == 1 and v[0]["chat_title"] == "Synthetic basic group"


def test_sink_admin_group_delete_revokes(tmp_path) -> None:
    c = FakeClient()
    s = sink(c, tmp_path, group_delete=True)
    e = run(s.handle_message(msg(BAD), BASIC_GROUP, SENDER, is_private=False))
    assert e["action"] == "delete_for_everyone"
    assert ("delete_messages", BASIC_GROUP, [10], True) in c.calls
    assert len(vault_tail(10, tmp_path / "vault.jsonl")) == 1


def test_sink_dry_run_no_client_calls_but_vaults(tmp_path) -> None:
    c = FakeClient()
    s = sink(c, tmp_path, dry_run=True)
    run(s.handle_message(msg(BAD, mid=1), DM, SENDER, is_private=True))
    run(s.handle_message(msg(GOOD, mid=2, sender_id=BLOCKED_ID), BASIC_GROUP, SENDER, is_private=False))
    assert c.calls == []
    v = vault_tail(10, tmp_path / "vault.jsonl")
    assert [x["result"] for x in v] == ["dry_run", "dry_run"] and v[1]["reason"] == "blocklist"


def test_sink_vault_line_for_every_nullified_case(tmp_path) -> None:
    cases = [({}, DM, True), ({}, BASIC_GROUP, False), ({"group_delete": True}, BASIC_GROUP, False),
             ({"dm_action": "mute"}, DM, True), ({"dm_action": "archive"}, DM, True),
             ({"dm_action": "log"}, DM, True), ({"dry_run": True}, DM, True),
             ({"group_action": "delete_for_me"}, SUPERGROUP, False)]
    for i, (cfg, chat, private) in enumerate(cases):
        s = sink(FakeClient(), tmp_path, **cfg)
        assert run(s.handle_message(msg(BAD, mid=100 + i), chat, SENDER, is_private=private)) is not None
    v = vault_tail(100, tmp_path / "vault.jsonl")
    assert len(v) == len(cases) and all(x["text"] == BAD for x in v)
    assert all(x["result"] != "pending" and not x["result"].startswith("error") for x in v)
    raw = (tmp_path / "vault.jsonl").read_text().splitlines()
    assert all(json.loads(line) for line in raw)


def test_sink_never_raises_and_vaults_on_action_error(tmp_path) -> None:
    class Broken(FakeClient):
        async def delete_messages(self, *a: Any, **k: Any) -> None:
            raise RuntimeError("synthetic failure")

    s = sink(Broken(), tmp_path)
    e = run(s.handle_message(msg(BAD), DM, SENDER, is_private=True))
    assert e["result"].startswith("error")
    assert vault_tail(10, tmp_path / "vault.jsonl")[0]["result"] == "error: RuntimeError"


def test_sink_ignores_outgoing_and_empty(tmp_path) -> None:
    c = FakeClient()
    s = sink(c, tmp_path)
    out = msg(BAD)
    out.out = True
    assert run(s.handle_message(out, DM, SENDER, is_private=True)) is None
    assert run(s.handle_message(msg("   "), DM, SENDER, is_private=True)) is None
    assert c.calls == [] and not (tmp_path / "vault.jsonl").exists()


def test_backfill_scores_unread(tmp_path) -> None:
    class Dialogs(FakeClient):
        async def iter_dialogs(self):
            for d in (SimpleNamespace(id=DM.id, entity=DM, unread_count=2, is_user=True),
                      SimpleNamespace(id=9, entity=BASIC_GROUP, unread_count=0, is_user=False)):
                yield d

        async def iter_messages(self, entity: Any, limit: int):
            self.calls.append(("iter_messages", entity, limit))
            for m in (msg(BAD, mid=1), msg(GOOD, mid=2), msg(BAD, mid=3)):
                yield m

    c = Dialogs()
    assert run(sink(c, tmp_path).backfill(0)) == 0
    assert run(sink(c, tmp_path).backfill(5)) == 2
    assert ("iter_messages", DM, 2) in c.calls
    assert [x["backfill"] for x in vault_tail(10, tmp_path / "vault.jsonl")] == [True, True]


def test_vault_cli_prints_tail(tmp_path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    path = tmp_path / "vault.jsonl"
    monkeypatch.setattr(tn, "VAULT_PATH", path)
    s = TelegramSink(FakeClient(), StubNullifier(), TgConfig(), vault_path=path, sleep=no_sleep)
    for i in range(3):
        run(s.handle_message(msg(BAD, mid=i), DM, SENDER, is_private=True))
    assert tn.main(["--vault", "2"]) == 0
    out = capsys.readouterr().out
    assert out.count("delete_for_me->deleted_for_me") == 2 and "msg=0" not in out and BAD in out


# ------------------------------------------------------------------------------------- author context (v2)

class AuthorStub(StubNullifier):
    """A v2 student stand-in: declares author/authors and records the author dicts it was given."""
    uses_author = True
    uses_author_bio = True
    text_model_enabled = True

    def __init__(self) -> None:
        super().__init__()
        self.authors: list[Any] = []

    def score(self, texts: list[str], authors: list[Any] | None = None) -> list[float]:
        return super().score(texts)

    def should_nullify(self, text: str, *, author_id: str | None = None, username: str | None = None,
                       author: dict | None = None) -> bool:
        self.authors.append(author)
        return super().should_nullify(text, author_id=author_id, username=username)


class BioClient(FakeClient):
    async def __call__(self, request: Any) -> Any:
        self.calls.append(("request", request))
        return SimpleNamespace(full_user=SimpleNamespace(about="synthetic bio text"))

    async def get_entity(self, entity: Any) -> Any:
        self.calls.append(("get_entity", entity))
        return SimpleNamespace(id=entity, username="synthetic_looked_up", first_name="Synthetic", verified=False)


def _requests(client: FakeClient) -> list[str]:
    return [type(c[1]).__name__ for c in client.calls if c[0] == "request"]


def test_author_context_fetched_lazily_once_per_sender(tmp_path) -> None:
    client, nul = BioClient(), AuthorStub()
    s = TelegramSink(client, nul, TgConfig(dm_action="log"), vault_path=tmp_path / "v.jsonl", sleep=no_sleep)
    user = SimpleNamespace(id=222, username="synthetic_sender", first_name="Synthetic", verified=True)
    for mid in (1, 2, 3):
        run(s.handle_message(msg(GOOD, mid=mid), DM, user, is_private=True))
    assert _requests(client) == ["GetFullUserRequest"]  # one bio lookup for three messages
    assert nul.authors[-1] == {"bio": "synthetic bio text", "followers": None, "following": None,
                               "tweet_count": None, "created_at": None, "verified": True}
    # sender missing from the event: one get_entity per sender id, cached
    for mid in (4, 5):
        run(s.handle_message(msg(GOOD, mid=mid, sender_id=333), DM, None, is_private=True))
    assert [c for c in client.names() if c == "get_entity"] == ["get_entity"]
    assert nul.calls[-1]["username"] == "synthetic_looked_up"
    assert _requests(client) == ["GetFullUserRequest", "GetFullUserRequest"]


def test_author_context_skipped_when_not_needed(tmp_path) -> None:
    client = BioClient()
    nul = AuthorStub()
    s = TelegramSink(client, nul, TgConfig(dm_action="log"), vault_path=tmp_path / "v.jsonl", sleep=no_sleep)
    blocked = SimpleNamespace(id=BLOCKED_ID, username=BLOCKED_NAME, first_name="S")
    run(s.handle_message(msg(GOOD, sender_id=BLOCKED_ID), DM, blocked, is_private=True))  # blocklist decides
    run(s.handle_message(msg("@x https://t.co/y", sender_id=444), DM, SENDER, is_private=True))  # low-info
    nul.text_model_enabled = False
    run(s.handle_message(msg(BAD, sender_id=555), DM, SENDER, is_private=True))
    assert _requests(client) == [] and all(a is None for a in nul.authors)
    # a channel posting in a group: verified only, no GetFullUser
    nul.text_model_enabled = True
    chan = SimpleNamespace(id=-100777, title="Synthetic channel", verified=False)
    run(s.handle_message(msg(GOOD, sender_id=-100777), BASIC_GROUP, chan, is_private=False))
    assert _requests(client) == [] and nul.authors[-1]["verified"] is False and nul.authors[-1]["bio"] is None
    # text-only deciders never get an author and never cause lookups
    plain_client = BioClient()
    plain = TelegramSink(plain_client, StubNullifier(), TgConfig(dm_action="log"), vault_path=tmp_path / "w.jsonl",
                         sleep=no_sleep)
    run(plain.handle_message(msg(BAD), DM, SimpleNamespace(username="u", first_name="S"), is_private=True))
    assert _requests(plain_client) == []


def test_bio_flood_wait_scores_without_bio(tmp_path) -> None:
    class Flooding(BioClient):
        async def __call__(self, request: Any) -> Any:
            self.calls.append(("request", request))
            raise FloodWaitError(request=None, capture=600)

    client, nul = Flooding(), AuthorStub()
    s = TelegramSink(client, nul, TgConfig(dm_action="log"), vault_path=tmp_path / "v.jsonl", sleep=no_sleep)
    entry = run(s.handle_message(msg(BAD), DM, SimpleNamespace(username="u", first_name="S"), is_private=True))
    assert entry is not None and entry["reason"] == "score"
    assert nul.authors[-1]["bio"] is None


def test_evaluate_passes_author_only_when_accepted() -> None:
    nul = AuthorStub()
    ctx = {"bio": "b", "verified": None}
    assert tn.evaluate(BAD, 222, "u", True, False, nul, TgConfig(), ctx)[0] is True
    assert nul.authors[-1] == ctx
    assert decide(BAD, 222, "u", True, False, StubNullifier(), TgConfig(), ctx) == (True, 0.9, "delete_for_me")


# ------------------------------------------------------------------------------ review fixes (synthetic)

import threading
import time

from telethon.errors import (AuthKeyDuplicatedError, AuthKeyUnregisteredError, MessageDeleteForbiddenError,
                             ServerError)
from telethon.tl.functions.users import GetFullUserRequest
from telethon.tl.types import InputPeerChannel, PeerChannel

from gvon import telegram_client as tgc


class CountingStub(StubNullifier):
    text_model_enabled = True

    def __init__(self) -> None:
        super().__init__()
        self.score_calls = 0
        self.threads: list[int] = []

    def score(self, texts: list[str]) -> list[float]:
        self.score_calls += 1
        return super().score(texts)

    def should_nullify(self, text: str, *, author_id: str | None = None, username: str | None = None) -> bool:
        self.threads.append(threading.get_ident())
        return self.is_blocked(author_id=author_id, username=username) or StubNullifier.score(self, [text])[0] >= 0.5


def test_score_only_for_nullified_and_classifier_off_loop(tmp_path) -> None:
    nul = CountingStub()
    s = TelegramSink(FakeClient(), nul, TgConfig(dm_action="log"), vault_path=tmp_path / "v.jsonl", sleep=no_sleep)
    for mid in range(5):
        assert run(s.handle_message(msg(GOOD, mid=mid), DM, SENDER, is_private=True)) is None
    assert nul.score_calls == 0  # kept messages are never scored
    assert threading.get_ident() not in nul.threads  # should_nullify ran in a worker thread
    run(s.handle_message(msg(BAD, mid=9), DM, SENDER, is_private=True))
    assert nul.score_calls == 1
    nul.text_model_enabled = False  # blocklist hit while text scoring is off: no score at all
    e = run(s.handle_message(msg(GOOD, mid=10, sender_id=BLOCKED_ID), DM, SENDER, is_private=True))
    assert nul.score_calls == 1 and e["score"] is None and e["reason"] == "blocklist"
    assert "score=-" in tn.format_vault_entry(e)


def test_mark_read_is_opt_in_safe_and_never_on_backfill(tmp_path) -> None:
    class Dialogs(FakeClient):
        unread = 1

        async def __call__(self, request: Any) -> Any:
            self.calls.append(("request", request))
            return SimpleNamespace(dialogs=[SimpleNamespace(unread_count=self.unread)])

    # default: a fallback mute never marks read
    c = Dialogs()
    run(apply("delete_for_me", c, SUPERGROUP, 9, is_channel=True, sleep=no_sleep))
    assert "send_read_acknowledge" not in c.names()
    # opted in: only when the flagged message is the chat's only unread message
    c = Dialogs()
    res = run(apply("delete_for_me", c, SUPERGROUP, 9, is_channel=True, mark_read=True, sleep=no_sleep))
    assert res.endswith("+ marked_read") and ("send_read_acknowledge", SUPERGROUP, 9) in c.calls
    c = Dialogs()
    c.unread = 3  # earlier legitimate unread messages: leave them unread
    res = run(apply("delete_for_me", c, SUPERGROUP, 9, is_channel=True, mark_read=True, sleep=no_sleep))
    assert "send_read_acknowledge" not in c.names() and "marked_read" not in res
    # sink: mark_read=True config, but backfill never marks read
    c = Dialogs()
    s = TelegramSink(c, StubNullifier(), TgConfig(group_action="delete_for_me", mark_read=True),
                     vault_path=tmp_path / "v.jsonl", sleep=no_sleep)
    run(s.handle_message(msg(BAD, mid=4), SUPERGROUP, SENDER, is_private=False, backfill=True))
    assert "send_read_acknowledge" not in c.names()
    # and a successful delete_for_me in a DM never marks read, even when opted in
    c = Dialogs()
    s = TelegramSink(c, StubNullifier(), TgConfig(mark_read=True), vault_path=tmp_path / "w.jsonl", sleep=no_sleep)
    run(s.handle_message(msg(BAD, mid=5), DM, SENDER, is_private=True))
    assert c.names() == ["delete_messages"]


def test_dm_delete_error_never_mutes_the_chat(tmp_path) -> None:
    class Failing(FakeClient):
        def __init__(self, exc: Exception) -> None:
            super().__init__()
            self.exc = exc

        async def delete_messages(self, *a: Any, **k: Any) -> None:
            raise self.exc

    c = Failing(ServerError(request=None, message="INTERNAL"))
    e = run(sink(c, tmp_path).handle_message(msg(BAD), DM, SENDER, is_private=True))
    assert e["result"] == "error: ServerError" and c.calls == []  # no mute, no read ack
    c = Failing(MessageDeleteForbiddenError(request=None))
    e = run(sink(c, tmp_path).handle_message(msg(BAD, mid=11), DM, SENDER, is_private=True))
    assert e["result"] == "error: MessageDeleteForbiddenError" and c.calls == []
    # basic group + permanent permission error: mute fallback; transient error: recorded, chat untouched
    c = Failing(MessageDeleteForbiddenError(request=None))
    e = run(sink(c, tmp_path, group_action="delete_for_me").handle_message(msg(BAD, mid=12), BASIC_GROUP, SENDER,
                                                                         is_private=False))
    assert e["result"].startswith("fallback_mute (MessageDeleteForbiddenError") and "request" in c.names()
    c = Failing(ServerError(request=None, message="INTERNAL"))
    e = run(sink(c, tmp_path, group_action="delete_for_me").handle_message(msg(BAD, mid=13), BASIC_GROUP, SENDER,
                                                                         is_private=False))
    assert e["result"] == "error: ServerError" and c.calls == []


def test_unresolved_chat_is_never_acted_on(tmp_path) -> None:
    c = FakeClient()
    m = msg(BAD, mid=40)
    m.chat_id = -1001234
    m.peer_id = PeerChannel(channel_id=1234)
    e = run(sink(c, tmp_path, group_action="delete_for_me").handle_message(m, None, SENDER, is_private=False))
    assert e["result"] == "skipped: unresolved chat" and c.calls == [] and e["chat_id"] == -1001234
    # an input peer of a supergroup (event.get_input_chat fallback) is treated as a channel: no delete-for-me
    c = FakeClient()
    peer = InputPeerChannel(channel_id=1234, access_hash=0)
    e = run(sink(c, tmp_path, group_action="delete_for_me").handle_message(m, peer, SENDER, is_private=False))
    assert e["result"].startswith("fallback_mute") and "delete_messages" not in c.names()

    class Event:
        is_private = False
        message = m

        async def get_chat(self):
            return None

        async def get_input_chat(self):
            return None

        async def get_sender(self):
            return SENDER

    c = FakeClient()
    run(sink(c, tmp_path, group_action="delete_for_me").on_new_message(Event()))
    assert c.calls == []


def test_service_support_and_allowlisted_senders_are_exempt(tmp_path) -> None:
    c = FakeClient()
    s = sink(c, tmp_path, allow=frozenset({"333", "synthetic_friend"}))
    service = SimpleNamespace(id=777000, first_name="Telegram")
    assert run(s.handle_message(msg(BAD, mid=50, sender_id=777000), service, None, is_private=True)) is None
    support = SimpleNamespace(username="synthetic_support", support=True)
    assert run(s.handle_message(msg(BAD, mid=51, sender_id=444), DM, support, is_private=True)) is None
    assert run(s.handle_message(msg(BAD, mid=52, sender_id=333), DM, SENDER, is_private=True)) is None
    friend = SimpleNamespace(username="Synthetic_Friend")
    assert run(s.handle_message(msg(BAD, mid=53, sender_id=555), DM, friend, is_private=True)) is None
    assert c.calls == [] and s.stats["exempt"] == 4 and not (tmp_path / "vault.jsonl").exists()


def test_delete_for_everyone_never_in_broadcast_channels_or_channel_posts(tmp_path) -> None:
    channel = SimpleNamespace(id=8001, title="Synthetic channel", broadcast=True)
    c = FakeClient()
    e = run(sink(c, tmp_path, group_delete=True).handle_message(msg(BAD), channel, SENDER, is_private=False))
    assert e["action"] == "log" and c.calls == []  # no admin lookup, no revoke
    forwarded = msg(BAD, mid=11)
    forwarded.fwd_from = SimpleNamespace(saved_from_peer=PeerChannel(channel_id=8001))
    c = FakeClient()
    e = run(sink(c, tmp_path, group_delete=True).handle_message(forwarded, BASIC_GROUP, SENDER, is_private=False))
    assert e["action"] == "log" and c.calls == []
    as_channel = SimpleNamespace(id=-1008001, title="Synthetic channel", broadcast=True, username=None)
    c = FakeClient()
    e = run(sink(c, tmp_path, group_delete=True).handle_message(msg(BAD, mid=12), BASIC_GROUP, as_channel,
                                                                is_private=False))
    assert e["action"] == "log" and c.calls == []


def test_media_is_saved_before_delete_or_message_is_only_logged(tmp_path) -> None:
    class MessageMediaPhoto:
        pass

    def photo_msg(mid: int, size: int = 15) -> SimpleNamespace:
        m = msg(BAD, mid=mid)
        m.media = MessageMediaPhoto()
        m.file = SimpleNamespace(mime_type="image/jpeg", size=size, name=None, id="synthetic-file-id")
        return m

    c = FakeClient()
    s = sink(c, tmp_path)
    e = run(s.handle_message(photo_msg(60), DM, SENDER, is_private=True))
    assert c.names() == ["download_media", "delete_messages"] and e["result"] == "deleted_for_me"
    saved = tmp_path / tn.MEDIA_DIR_NAME / str(DM.id) / "60" / "synthetic.jpg"
    assert saved.read_bytes() == b"synthetic-bytes" and saved.stat().st_mode & 0o777 == 0o600
    v = vault_tail(1, tmp_path / "vault.jsonl")[0]
    assert v["media"]["type"] == "photo" and v["media"]["file_id"] == "synthetic-file-id"
    assert v["media"]["path"].endswith("synthetic.jpg")

    class NoDownload(FakeClient):
        async def download_media(self, message: Any, file: Any = None) -> str:
            raise RuntimeError("synthetic download failure")

    c = NoDownload()
    e = run(sink(c, tmp_path).handle_message(photo_msg(61), DM, SENDER, is_private=True))
    assert e["result"] == "logged (media not saved: RuntimeError)" and c.calls == []
    c = FakeClient()
    e = run(sink(c, tmp_path, media_max_mb=0.001).handle_message(photo_msg(62, size=10_000), DM, SENDER,
                                                               is_private=True))
    assert e["result"].startswith("logged (media not saved: larger than") and c.calls == []
    # mute keeps the message, so no download is needed
    c = FakeClient()
    run(sink(c, tmp_path, dm_action="mute").handle_message(photo_msg(63), DM, SENDER, is_private=True))
    assert "download_media" not in c.names()


def test_bio_lookup_is_bounded_and_skipped_while_flood_limited(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tn, "BIO_TIMEOUT_S", 0.05)

    class Slow(BioClient):  # what Telethon does inside a <= 60 s flood wait: sleep in the call
        async def __call__(self, request: Any) -> Any:
            self.calls.append(("request", request))
            await asyncio.sleep(5)

    client, nul = Slow(), AuthorStub()
    s = TelegramSink(client, nul, TgConfig(dm_action="log"), vault_path=tmp_path / "v.jsonl", sleep=no_sleep)
    user = SimpleNamespace(id=222, username="u", first_name="S")
    t0 = time.perf_counter()
    for mid in (1, 2):
        run(s.handle_message(msg(GOOD, mid=mid), DM, user, is_private=True))
    assert time.perf_counter() - t0 < 2 and nul.authors[-1]["bio"] is None
    assert _requests(client) == ["GetFullUserRequest"]  # the timed-out sender is not retried per message
    # while Telethon has GetFullUser flood-limited, no request is sent at all
    client2, nul2 = BioClient(), AuthorStub()
    client2._flood_waited_requests = {GetFullUserRequest.CONSTRUCTOR_ID: time.time() + 30}
    s2 = TelegramSink(client2, nul2, TgConfig(dm_action="log"), vault_path=tmp_path / "w.jsonl", sleep=no_sleep)
    run(s2.handle_message(msg(GOOD, mid=3), DM, user, is_private=True))
    assert _requests(client2) == [] and nul2.authors[-1]["bio"] is None


def test_reconnect_loop_exits_on_revoked_session_and_backs_off() -> None:
    class Conn:
        def __init__(self, errors: list[Exception | None]) -> None:
            self.errors = errors
            self.connects = 0

        async def run_until_disconnected(self) -> None:
            e = self.errors.pop(0)
            if e is not None:
                raise e

        async def connect(self) -> None:
            self.connects += 1

    slept: list[float] = []

    async def rec(s: float) -> None:
        slept.append(s)

    c = Conn([AuthKeyUnregisteredError(request=None)])
    with pytest.raises(tgc.TelegramConfigError, match="make telegram-login"):
        run(tn.reconnect_loop(c, sleep=rec))
    assert slept == [] and c.connects == 0
    with pytest.raises(tgc.TelegramConfigError):
        run(tn.reconnect_loop(Conn([AuthKeyDuplicatedError(request=None)]), sleep=rec))
    # quick drops back off exponentially even though connect() succeeds
    c = Conn([RuntimeError("x"), None, RuntimeError("y"), AuthKeyUnregisteredError(request=None)])
    with pytest.raises(tgc.TelegramConfigError):
        run(tn.reconnect_loop(c, sleep=rec, clock=lambda: 0.0))
    assert slept == [5.0, 10.0, 20.0] and c.connects == 3
    # a connection that stayed up resets the backoff
    slept.clear()
    ticks = iter([0.0, 0.0, 0.0, 1000.0, 1000.0, 1000.0])
    c = Conn([None, None, AuthKeyUnregisteredError(request=None)])
    with pytest.raises(tgc.TelegramConfigError):
        run(tn.reconnect_loop(c, sleep=rec, clock=lambda: next(ticks)))
    assert slept == [5.0, 5.0]
