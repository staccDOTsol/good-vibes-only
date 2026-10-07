"""Unit tests for gvon.telegram_ingest with fake Telethon objects (plain classes, no network, no session).

Every id, name, title and text here is synthetic. The fakes mirror only the attributes the collector
reads from Telethon's User / Chat / Channel / Message / Dialog objects.
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from gvon import telegram_ingest as ti

NOW = datetime(2026, 10, 6, 12, 0, tzinfo=timezone.utc)
ME_ID = 500


def ago(days: float) -> datetime:
    return NOW - timedelta(days=days)


# --------------------------------------------------------------------------- fakes


class FloodWaitError(Exception):
    """Same name and `seconds` attribute as telethon.errors.FloodWaitError."""

    def __init__(self, seconds: int) -> None:
        super().__init__(f"synthetic flood wait {seconds}")
        self.seconds = seconds


class User:
    def __init__(self, id: int, username: str | None = None, first_name: str = "", last_name: str | None = None,
                 bot: bool = False, is_self: bool = False, scam: bool = False) -> None:
        self.id, self.username, self.first_name, self.last_name = id, username, first_name, last_name
        self.bot, self.is_self, self.scam, self.fake, self.verified, self.premium = bot, is_self, scam, False, False, False


class Chat:  # basic group: no bot / broadcast / megagroup attributes
    def __init__(self, id: int, title: str) -> None:
        self.id, self.title = id, title


class Channel:
    def __init__(self, id: int, title: str, megagroup: bool = False, broadcast: bool = False, creator: bool = False,
                 admin_rights: Any = None, username: str | None = None) -> None:
        self.id, self.title, self.megagroup, self.broadcast = id, title, megagroup, broadcast
        self.creator, self.admin_rights, self.username, self.gigagroup = creator, admin_rights, username, False


class ReplyHeader:
    def __init__(self, reply_to_msg_id: int, forum_topic: bool = False, reply_to_top_id: int | None = None,
                 reply_to_peer_id: Any = None) -> None:
        self.reply_to_msg_id, self.forum_topic = reply_to_msg_id, forum_topic
        self.reply_to_top_id, self.reply_to_peer_id = reply_to_top_id, reply_to_peer_id


class MessageMediaPhoto:
    pass


class Msg:
    def __init__(self, id: int, date: datetime, text: str = "", sender: Any = None, sender_id: int | None = None,
                 out: bool = False, mentioned: bool = False, reply_to: int | ReplyHeader | None = None,
                 media: Any = None, action: Any = None) -> None:
        self.id, self.date, self.message, self.out, self.mentioned = id, date, text, out, mentioned
        self.sender = sender
        self.sender_id = sender_id if sender_id is not None else (getattr(sender, "id", None) if sender else None)
        self.reply_to = ReplyHeader(reply_to) if isinstance(reply_to, int) else reply_to
        self.media, self.action = media, action


class Dialog:
    def __init__(self, id: int, entity: Any, date: datetime, name: str | None = None, top: int = 0,
                 pinned: bool = False) -> None:
        self.id, self.entity, self.date, self.pinned = id, entity, date, pinned
        self.name = name if name is not None else (getattr(entity, "title", None) or getattr(entity, "first_name", ""))
        self.message = SimpleNamespace(id=top)


class FakeAPI:
    """Implements gvon.telegram_ingest.TelegramAPI over in-memory histories keyed by entity id."""

    def __init__(self, me: User, dialogs: list[Dialog], history: dict[int, list[Msg]],
                 linked: dict[int, tuple[int, Any]] | None = None, flood: dict[int, list[int]] | None = None,
                 no_mention_search: bool = False) -> None:
        self._me, self._dialogs, self.history = me, dialogs, history
        self.linked = linked or {}
        self.flood = flood or {}  # entity id -> flood seconds to raise on the next history call(s)
        self.no_mention_search = no_mention_search
        self.calls: list[tuple[str, int, dict[str, Any]]] = []

    async def me(self) -> User:
        return self._me

    async def _iter(self, items: list[Any]):
        for x in items:
            yield x

    def dialogs(self):
        return self._iter(self._dialogs)

    def messages(self, entity: Any, *, min_id: int = 0, from_me: bool = False, mentions: bool = False):
        self.calls.append(("messages", entity.id, {"min_id": min_id, "from_me": from_me, "mentions": mentions}))
        pending = self.flood.get(entity.id)
        if pending:
            secs = pending.pop(0)

            async def boom():
                raise FloodWaitError(secs)
                yield  # pragma: no cover

            return boom()
        if mentions and self.no_mention_search:
            async def unsupported():
                raise RuntimeError("synthetic: search filter not supported")
                yield  # pragma: no cover

            return unsupported()
        msgs = sorted(self.history.get(entity.id, []), key=lambda m: -m.id)
        msgs = [m for m in msgs if m.id > min_id]
        if from_me:
            msgs = [m for m in msgs if m.out]
        if mentions:
            msgs = [m for m in msgs if m.mentioned]
        return self._iter(msgs)

    async def get_messages(self, entity: Any, ids: list[int]) -> list[Any]:
        self.calls.append(("get_messages", entity.id, {"ids": list(ids)}))
        by_id = {m.id: m for m in self.history.get(entity.id, [])}
        return [by_id.get(i) for i in ids]

    async def linked_chat(self, entity: Any) -> tuple[int, Any] | None:
        self.calls.append(("linked_chat", entity.id, {}))
        return self.linked.get(entity.id)


# --------------------------------------------------------------------------- a synthetic account


ME = User(ME_ID, "synth_builder", "Synth", is_self=True)
FRIEND = User(601, "synth_friend", "Friend")
BEGGAR = User(602, None, "Beggar")
FUDDER = User(603, "synth_fudder", "Fudder", scam=True)
PINGER = User(604, "synth_pinger", "Pinger")
CHATTER = User(605, "synth_chatter", "Chatter")
COMMENTER = User(606, "synth_commenter", "Commenter")
SERVICE = User(777000, None, "Telegram")
TRADEBOT = User(700, "synth_bot", "Bot", bot=True)
ACTIVE = Channel(1001, "Synthetic Builders", megagroup=True)  # you posted here this week
QUIET = Channel(1002, "Synthetic Quiet Group", megagroup=True)  # you did not post here this week
BASIC = Chat(77, "Synthetic Basic Group")
NEWS = Channel(1003, "Synthetic News", broadcast=True)  # someone else's channel
MINE = Channel(1004, "Synthetic Announcements", broadcast=True, creator=True)
DISCUSSION = Channel(1005, "Synthetic Announcements Chat", megagroup=True)
SAVED = User(ME_ID, "synth_builder", "Synth", is_self=True)
STALE = User(608, "synth_stale", "Stale")

ACTIVE_PEER, QUIET_PEER, BASIC_PEER = -1001001, -1001002, -77
MINE_PEER, DISCUSSION_PEER, NEWS_PEER = -1001004, -1001005, -1001003


def build_api(**kw: Any) -> FakeAPI:
    history = {
        FRIEND.id: [Msg(1, ago(1), "gm, call later?", sender=FRIEND), Msg(2, ago(0.9), "sure", sender=ME, out=True),
                    Msg(3, ago(0.5), "ok", sender=FRIEND, reply_to=2), Msg(0, ago(9), "too old", sender=FRIEND)],
        BEGGAR.id: [Msg(5, ago(2), "sir please send gas sir", sender=None, sender_id=None),
                    Msg(6, ago(2), "", sender=BEGGAR, media=MessageMediaPhoto())],
        ACTIVE.id: [
            Msg(10, ago(3), "we ship the router today", sender=ME, out=True),
            Msg(11, ago(2.5), "rug incoming, devs dump", sender=FUDDER, reply_to=10),
            Msg(12, ago(2.4), "@synth_builder wen airdrop", sender=PINGER, mentioned=True),
            Msg(13, ago(2.3), "anyone tried the new wallet?", sender=CHATTER),
            Msg(14, ago(2.2), "", sender=CHATTER, action=SimpleNamespace()),  # service message
            Msg(15, ago(2.1), "replying to your old post", sender=PINGER, mentioned=True, reply_to=4),
            Msg(4, ago(20), "my old post", sender=ME, out=True),
            Msg(16, ago(2.0), "topic chatter", sender=CHATTER, reply_to=ReplyHeader(9, forum_topic=True)),
        ],
        QUIET.id: [
            Msg(20, ago(30), "my quiet old post", sender=ME, out=True),
            Msg(21, ago(1), "random chatter", sender=CHATTER),
            Msg(22, ago(1), "this guy is a scammer", sender=FUDDER, mentioned=True, reply_to=20),
            Msg(23, ago(1), "hey @synth_builder", sender=PINGER, mentioned=True),
        ],
        BASIC.id: [Msg(30, ago(1), "basic group chatter", sender=CHATTER)],
        DISCUSSION.id: [
            Msg(40, ago(1), "launch post", sender=MINE, sender_id=MINE_PEER),
            Msg(41, ago(0.5), "great launch", sender=COMMENTER, reply_to=40),
        ],
        SERVICE.id: [Msg(50, ago(0.1), "Login code: 00000", sender=SERVICE)],
        SAVED.id: [Msg(60, ago(0.1), "note to self", sender=ME, out=True)],
        TRADEBOT.id: [Msg(70, ago(0.2), "your order filled", sender=TRADEBOT)],
        STALE.id: [Msg(80, ago(12), "old dm", sender=STALE)],
    }
    dialogs = [
        Dialog(SAVED.id, SAVED, ago(0.1), pinned=True),
        Dialog(SERVICE.id, SERVICE, ago(0.1)),
        Dialog(TRADEBOT.id, TRADEBOT, ago(0.2), top=70),
        Dialog(FRIEND.id, FRIEND, ago(0.5), top=3),
        Dialog(DISCUSSION_PEER, DISCUSSION, ago(0.5), top=41),
        Dialog(QUIET_PEER, QUIET, ago(1), top=23),
        Dialog(BASIC_PEER, BASIC, ago(1), top=30),
        Dialog(BEGGAR.id, BEGGAR, ago(2), top=6),
        Dialog(ACTIVE_PEER, ACTIVE, ago(2), top=16),
        Dialog(NEWS_PEER, NEWS, ago(1)),
        Dialog(MINE_PEER, MINE, ago(1)),
        Dialog(STALE.id, STALE, ago(12), top=80),
    ]
    return FakeAPI(ME, dialogs, history, linked={MINE.id: (DISCUSSION_PEER, DISCUSSION)}, **kw)


def run_pull(api: FakeAPI, out: Path, **kw: Any) -> tuple[dict, list[float]]:
    slept: list[float] = []

    async def fake_sleep(s: float) -> None:
        slept.append(s)

    summary = asyncio.run(ti.pull(api, out_dir=out, now=kw.pop("now", NOW), sleep=fake_sleep, **kw))
    return summary, slept


def read_rows(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


# --------------------------------------------------------------------------- pure helpers


def test_chat_type_and_ids() -> None:
    assert ti.chat_type(FRIEND) == "private" and ti.chat_type(TRADEBOT) == "bot"
    assert ti.chat_type(BASIC) == "group" and ti.chat_type(ACTIVE) == "supergroup"
    assert ti.chat_type(NEWS) == "channel"
    assert ti.msg_key(-1001001, 11) == "-1001001:11"
    assert ti.iso(datetime(2026, 10, 1, 1, 2, 3)) == "2026-10-01T01:02:03Z"
    assert ti.display_name(FRIEND) == "Friend" and ti.display_name(ACTIVE) == "Synthetic Builders"
    assert ti.media_kind(Msg(1, NOW, media=MessageMediaPhoto())) == "photo"


def test_reply_target_handles_forum_topics_and_foreign_peers() -> None:
    assert ti.reply_target(Msg(1, NOW)) is None
    assert ti.reply_target(Msg(1, NOW, reply_to=7)) == 7
    assert ti.reply_target(Msg(1, NOW, reply_to=ReplyHeader(9, forum_topic=True))) is None  # topic root, not a reply
    assert ti.reply_target(Msg(1, NOW, reply_to=ReplyHeader(8, forum_topic=True, reply_to_top_id=9))) == 8
    assert ti.reply_target(Msg(1, NOW, reply_to=ReplyHeader(8, reply_to_peer_id=object()))) is None


def test_classify_kinds() -> None:
    own = {ME_ID}
    c = lambda m, ctype="supergroup", posted=False, rs=None: ti.classify(  # noqa: E731
        m, ctype=ctype, own_sender_ids=own, own_msg_ids={10}, posted=posted, reply_sender_id=rs)
    assert c(Msg(1, NOW, sender=ME, out=True)) == "own"
    assert c(Msg(1, NOW, sender=FRIEND), ctype="private") == "dm"
    assert c(Msg(1, NOW, sender=TRADEBOT), ctype="bot") == "dm"
    assert c(Msg(1, NOW, sender=FUDDER, reply_to=10)) == "group_reply"
    assert c(Msg(1, NOW, sender=FUDDER, reply_to=99), rs=ME_ID) == "group_reply"
    assert c(Msg(1, NOW, sender=PINGER, mentioned=True)) == "group_mention"
    assert c(Msg(1, NOW, sender=CHATTER), posted=True) == "group"
    assert c(Msg(1, NOW, sender=CHATTER)) is None
    # your channel's posts count as yours
    assert ti.classify(Msg(1, NOW, sender_id=MINE_PEER), ctype="supergroup", own_sender_ids={ME_ID, MINE_PEER},
                       own_msg_ids=set(), posted=True) == "own"


def test_apply_cap_keeps_replies_first_and_counts_drops() -> None:
    cands = [(Msg(i, NOW), "group") for i in range(1, 6)] + [(Msg(10, NOW), "group_reply"), (Msg(11, NOW), "own")]
    kept, dropped = ti.apply_cap(cands, 3)
    assert [(m.id, k) for m, k in kept] == [(5, "group"), (10, "group_reply"), (11, "own")]
    assert dropped == {"group": 4}


def test_flood_wait_seconds_duck_types_telethon_errors() -> None:
    assert ti.flood_wait_seconds(FloodWaitError(42)) == 42
    assert ti.flood_wait_seconds(RuntimeError("x")) is None
    from telethon.errors import FloodWaitError as RealFloodWait

    assert ti.flood_wait_seconds(RealFloodWait(None, capture=7)) == 7


# --------------------------------------------------------------------------- pull


def test_pull_collects_kinds_rows_and_users(tmp_path: Path) -> None:
    api = build_api()
    summary, slept = run_pull(api, tmp_path)
    assert slept == []
    rows = read_rows(tmp_path / ti.TWEETS_NAME)
    by_id = {r["id"]: r for r in rows}
    kinds = {rid: r["kind"] for rid, r in by_id.items()}
    assert kinds == {
        "601:1": "dm", "601:2": "own", "601:3": "dm",
        "602:5": "dm", "602:6": "dm",
        f"{ACTIVE_PEER}:10": "own", f"{ACTIVE_PEER}:11": "group_reply", f"{ACTIVE_PEER}:12": "group_mention",
        f"{ACTIVE_PEER}:13": "group", f"{ACTIVE_PEER}:15": "group_reply", f"{ACTIVE_PEER}:16": "group",
        f"{QUIET_PEER}:22": "group_reply", f"{QUIET_PEER}:23": "group_mention",
        f"{DISCUSSION_PEER}:40": "own", f"{DISCUSSION_PEER}:41": "group_reply",
        "700:70": "dm",
    }
    # never collected: Telegram service (login codes), Saved Messages, someone else's broadcast channel,
    # chatter in groups you did not post in, service messages, messages older than the window
    assert not any(r["id"].startswith(("777000:", f"{NEWS_PEER}:", f"{BASIC_PEER}:", "608:")) for r in rows)
    assert all("Login code" not in r["text"] for r in rows)
    assert f"{ACTIVE_PEER}:14" not in by_id and "601:0" not in by_id and f"{QUIET_PEER}:21" not in by_id

    fud = by_id[f"{ACTIVE_PEER}:11"]
    tweet_keys = {"id", "text", "author_id", "author_username", "author_name", "created_at", "conversation_id",
                  "in_reply_to_user_id", "lang", "kind", "referenced", "public_metrics"}
    assert tweet_keys <= set(fud) and {"platform", "chat_title", "chat_type"} <= set(fud)
    assert fud["author_id"] == "603" and fud["author_username"] == "synth_fudder" and fud["author_name"] == "Fudder"
    assert fud["conversation_id"] == str(ACTIVE_PEER) and fud["in_reply_to_user_id"] == str(ME_ID)
    assert fud["referenced"] == [{"type": "replied_to", "id": f"{ACTIVE_PEER}:10"}]
    assert fud["lang"] is None and fud["public_metrics"] == {} and fud["platform"] == "telegram"
    assert fud["chat_title"] == "Synthetic Builders" and fud["chat_type"] == "supergroup"
    assert fud["created_at"] == "2026-10-04T00:00:00Z"
    assert by_id[f"{ACTIVE_PEER}:16"]["referenced"] == []  # forum topic root is not a reply
    beg = by_id["602:5"]  # DM sender filled from the dialog when Telethon gives no sender
    assert beg["author_id"] == "602" and beg["author_username"] is None and beg["chat_type"] == "private"
    assert by_id["602:6"]["media"] == "photo" and by_id["602:6"]["text"] == ""
    assert by_id["700:70"]["chat_type"] == "bot"
    assert by_id[f"{DISCUSSION_PEER}:41"]["in_reply_to_user_id"] == str(MINE_PEER)

    # replied-to messages outside the collected rows become context (fetched in one batch per chat)
    ctx = {r["id"]: r for r in read_rows(tmp_path / ti.CONTEXT_NAME)}
    assert set(ctx) == {f"{ACTIVE_PEER}:4", f"{QUIET_PEER}:20"}
    assert ctx[f"{QUIET_PEER}:20"]["kind"] == "context" and ctx[f"{QUIET_PEER}:20"]["text"] == "my quiet old post"
    # the quiet group was read through the mentions search only, never scanned in full
    quiet_calls = [c[2] for c in api.calls if c[0] == "messages" and c[1] == QUIET.id]
    assert quiet_calls == [{"min_id": 0, "from_me": True, "mentions": False},
                           {"min_id": 0, "from_me": False, "mentions": True}]

    users = {u["id"]: u for u in read_rows(tmp_path / ti.USERS_NAME)}
    assert users[str(ME_ID)]["is_self"] is True and users[str(ME_ID)]["username"] == "synth_builder"
    assert users["603"]["scam"] is True and users["603"]["platform"] == "telegram" and users["603"]["description"] is None
    assert users["700"]["bot"] is True
    assert {"601", "602", "603", "604", "605", "606", "700"} <= set(users)

    assert summary["chats_skipped"] == {"saved_messages": 1, "telegram_service": 1, "broadcast_channel": 1,
                                        "inactive_in_window": 1}
    assert summary["chat_modes"] == {"dm": 3, "scan": 2, "mentions": 2}
    state = json.loads((tmp_path / ti.STATE_NAME).read_text())
    assert state["account_id"] == ME_ID
    assert state["chats"][str(ACTIVE_PEER)]["last_id"] == 16
    assert state["chats"][str(QUIET_PEER)]["last_id"] == 23
    assert state["chats"]["601"]["last_id"] == 3
    assert json.loads((tmp_path / ti.SUMMARY_NAME).read_text())["rows_total"] == len(rows)


def test_pull_resumes_from_state_and_merges(tmp_path: Path) -> None:
    api = build_api()
    run_pull(api, tmp_path)
    first = {r["id"] for r in read_rows(tmp_path / ti.TWEETS_NAME)}
    api.history[FRIEND.id].append(Msg(4, ago(0.05), "new message", sender=FRIEND))
    api.calls.clear()
    summary, _ = run_pull(api, tmp_path)
    friend_calls = [c[2] for c in api.calls if c[0] == "messages" and c[1] == FRIEND.id]
    assert friend_calls == [{"min_id": 3, "from_me": False, "mentions": False}]
    active_calls = [c[2]["min_id"] for c in api.calls if c[0] == "messages" and c[1] == ACTIVE.id]
    assert active_calls and set(active_calls) == {16}
    rows = {r["id"] for r in read_rows(tmp_path / ti.TWEETS_NAME)}
    assert rows == first | {"601:4"}
    assert summary["rows_this_run"] == {"dm": 1}
    assert json.loads((tmp_path / ti.STATE_NAME).read_text())["chats"]["601"]["last_id"] == 4
    # rows that age out of the window are dropped on the next flush
    run_pull(api, tmp_path, now=NOW + timedelta(days=4.6))  # window now starts 2.4 days ago
    later = {r["id"] for r in read_rows(tmp_path / ti.TWEETS_NAME)}
    assert f"{ACTIVE_PEER}:11" not in later and "601:4" in later
    # --fresh ignores state
    api.calls.clear()
    run_pull(api, tmp_path, fresh=True)
    assert {c[2]["min_id"] for c in api.calls if c[0] == "messages"} == {0}


def test_state_of_another_account_is_discarded(tmp_path: Path) -> None:
    (tmp_path / ti.STATE_NAME).write_text(json.dumps({"version": ti.STATE_VERSION, "account_id": 999,
                                                      "chats": {"601": {"last_id": 100}}}))
    api = build_api()
    run_pull(api, tmp_path)
    assert {c[2]["min_id"] for c in api.calls if c[0] == "messages" and c[1] == FRIEND.id} == {0}
    assert json.loads((tmp_path / ti.STATE_NAME).read_text())["account_id"] == ME_ID


def test_flood_wait_sleeps_requested_seconds_capped(tmp_path: Path) -> None:
    api = build_api(flood={FRIEND.id: [42], ACTIVE.id: [5000]})
    summary, slept = run_pull(api, tmp_path)
    assert slept == [42, ti.FLOOD_WAIT_CAP_S]
    kinds = {r["id"]: r["kind"] for r in read_rows(tmp_path / ti.TWEETS_NAME)}
    assert kinds["601:1"] == "dm" and kinds[f"{ACTIVE_PEER}:11"] == "group_reply"  # retried chats completed


def test_flood_wait_gives_up_after_max_retries_but_keeps_finished_chats(tmp_path: Path) -> None:
    api = build_api(flood={ACTIVE.id: [1] * (ti.MAX_FLOOD_RETRIES + 1)})
    with pytest.raises(FloodWaitError):
        run_pull(api, tmp_path)
    state = json.loads((tmp_path / ti.STATE_NAME).read_text())
    assert "601" in state["chats"] and str(ACTIVE_PEER) not in state["chats"]
    assert "601:1" in {r["id"] for r in read_rows(tmp_path / ti.TWEETS_NAME)}


def test_cap_chats_bots_and_mention_fallback(tmp_path: Path) -> None:
    api = build_api(no_mention_search=True)
    summary, _ = run_pull(api, tmp_path, max_per_chat=2, skip_bots=True)
    rows = read_rows(tmp_path / ti.TWEETS_NAME)
    per_chat: dict[str, list[str]] = {}
    for r in rows:
        per_chat.setdefault(r["conversation_id"], []).append(r["kind"])
    assert all(len(v) <= 2 for v in per_chat.values())
    assert sorted(per_chat[str(ACTIVE_PEER)]) == ["group_reply", "group_reply"]
    assert summary["dropped_by_cap_this_run"]["group"] >= 2
    assert "700" not in per_chat and summary["chats_skipped"]["bot"] == 1
    assert summary["chat_modes"].get("mentions_scan") == 2  # quiet + basic groups fell back to a scan
    assert {r["kind"] for r in rows if r["conversation_id"] == str(QUIET_PEER)} == {"group_reply", "group_mention"}
    # --chats N limits how many chats are pulled (most recent first)
    other = tmp_path / "limited"
    summary, _ = run_pull(build_api(), other, max_chats=1)
    assert summary["chats_pulled"] == 1
    assert {r["conversation_id"] for r in read_rows(other / ti.TWEETS_NAME)} == {"700"}


def test_api_credentials_never_echo_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TG_API_ID", "")
    monkeypatch.setenv("TG_API_HASH", "")
    with pytest.raises(ti.ConfigError):
        ti.api_credentials()
    monkeypatch.setenv("TG_API_ID", "not-a-number-synthetic")
    monkeypatch.setenv("TG_API_HASH", "synthetic-hash")
    with pytest.raises(ti.ConfigError) as e:
        ti.api_credentials()
    assert "synthetic" not in str(e.value)
    monkeypatch.setenv("TG_API_ID", "12345")
    assert ti.api_credentials() == (12345, "synthetic-hash")


def test_session_prefers_string_session(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from gvon import telegram_client as tgc

    monkeypatch.setenv("TG_SESSION", "")
    assert tgc.resolve_session(tmp_path / "s" / "telegram.session") == str(tmp_path / "s" / "telegram.session")
    assert (tmp_path / "s").is_dir()
    assert not tgc.has_session(tmp_path / "s" / "telegram.session")
    from telethon.crypto import AuthKey
    from telethon.sessions import StringSession

    synthetic = StringSession()
    synthetic.set_dc(2, "127.0.0.1", 443)
    synthetic.auth_key = AuthKey(b"\x01" * 256)  # synthetic key, not a real login
    monkeypatch.setenv("TG_SESSION", synthetic.save())
    assert isinstance(tgc.resolve_session(tmp_path / "x.session"), StringSession)
    assert isinstance(tgc.resolve_session(tmp_path / "x.session", allow_string=False), str)  # login always uses the file
    assert tgc.has_session(tmp_path / "missing.session")


def test_single_client_factory(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Ingest (login + pull) and the sink all build their TelegramClient through gvon.telegram_client.make_client."""
    import telethon

    from gvon import telegram_client as tgc
    from gvon import telegram_nullifier as tn

    assert not hasattr(ti, "make_client") and not hasattr(ti, "make_session")
    assert not hasattr(tn, "make_client")  # imported lazily from gvon.telegram_client inside run_sink
    built: list[tuple[Any, int, str, dict[str, Any]]] = []

    class FakeTelegramClient:
        def __init__(self, session: Any, api_id: int, api_hash: str, **kw: Any) -> None:
            built.append((session, api_id, api_hash, kw))

    monkeypatch.setattr(telethon, "TelegramClient", FakeTelegramClient)
    monkeypatch.setenv("TG_API_ID", "12345")
    monkeypatch.setenv("TG_API_HASH", "synthetic-hash")
    monkeypatch.setenv("TG_SESSION", "")
    sess = tmp_path / "d" / "telegram.session"
    tgc.make_client(sess)
    tgc.make_client(sess, allow_string=False, flood_sleep_threshold=ti.TELETHON_FLOOD_SLEEP_S)
    assert [b[0] for b in built] == [str(sess), str(sess)]
    assert all(b[1:3] == (12345, "synthetic-hash") for b in built)
    assert built[0][3]["flood_sleep_threshold"] == 60
    assert (tmp_path / "d").stat().st_mode & 0o777 == 0o700


def test_has_creds_never_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    from gvon import telegram_client as tgc

    monkeypatch.setenv("TG_API_ID", "")
    monkeypatch.setenv("TG_API_HASH", "")
    assert tgc.has_creds() is False
    monkeypatch.setenv("TG_API_ID", "not-an-int-synthetic")
    monkeypatch.setenv("TG_API_HASH", "synthetic-hash")
    assert tgc.has_creds() is False
    monkeypatch.setenv("TG_API_ID", "12345")
    assert tgc.has_creds() is True


def test_default_paths_are_under_gitignored_data() -> None:
    assert ti.DEFAULT_SESSION.parent.name == "data" and ti.DEFAULT_OUT.parts[-2:] == ("data", "raw")


def test_readiness_skips_without_creds_or_session(monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
                                                  capsys: pytest.CaptureFixture[str]) -> None:
    from gvon import telegram_client as tgc

    sess = tmp_path / "telegram.session"
    monkeypatch.setenv("TG_SESSION", "")
    monkeypatch.setenv("TG_API_ID", "")
    monkeypatch.setenv("TG_API_HASH", "")
    assert tgc.main(["ready", "--session", str(sess)]) == 3
    assert "TG_API_ID" in capsys.readouterr().err
    monkeypatch.setenv("TG_API_ID", "12345")
    monkeypatch.setenv("TG_API_HASH", "synthetic-hash-value")
    assert tgc.main(["ready", "--session", str(sess)]) == 3
    err = capsys.readouterr().err
    assert "make telegram-login" in err and "synthetic-hash-value" not in err
    sess.write_bytes(b"")  # synthetic placeholder file, not a real session
    assert tgc.main(["ready", "--session", str(sess)]) == 0
    out = capsys.readouterr().out
    assert "ready" in out and "synthetic-hash-value" not in out and "12345" not in out


# --------------------------------------------------------------------------- review fixes (synthetic)


def test_one_inaccessible_chat_does_not_abort_the_pull(tmp_path: Path) -> None:
    from telethon.errors import ChannelPrivateError

    class Banned(FakeAPI):
        def messages(self, entity: Any, **kw: Any):
            if entity.id == QUIET.id:
                async def boom():
                    raise ChannelPrivateError(request=None)
                    yield  # pragma: no cover

                return boom()
            return super().messages(entity, **kw)

    base = build_api()
    api = Banned(base._me, base._dialogs, base.history, linked=base.linked)
    for _ in range(2):
        summary, _ = run_pull(api, tmp_path)
        assert summary["chats_failed"] == {"ChannelPrivateError": 1}
    state = json.loads((tmp_path / ti.STATE_NAME).read_text())
    assert str(QUIET_PEER) not in state["chats"]  # retried next run
    assert {str(BASIC_PEER), "602", str(ACTIVE_PEER)} <= set(state["chats"])  # chats after it still collected


def test_forbidden_dialogs_are_skipped(tmp_path: Path) -> None:
    class ChannelForbidden:
        def __init__(self, id: int, title: str) -> None:
            self.id, self.title, self.megagroup, self.broadcast = id, title, True, False

    base = build_api()
    gone = ChannelForbidden(1009, "Synthetic banned group")
    dialogs = base._dialogs + [Dialog(-1001009, gone, None)]
    summary, _ = run_pull(FakeAPI(base._me, dialogs, base.history, linked=base.linked), tmp_path)
    assert summary["chats_skipped"]["forbidden"] == 1 and summary["chats_failed"] == {}


def test_session_paths_follow_telethon_suffix_rule(tmp_path: Path) -> None:
    import os

    from gvon import telegram_client as tgc

    scratch = tmp_path / "scratch"
    scratch.mkdir(mode=0o755)
    os.chmod(scratch, 0o755)
    assert tgc.resolve_session(scratch / "mytg") == str(scratch / "mytg.session")
    assert scratch.stat().st_mode & 0o777 == 0o755  # an existing arbitrary parent is never chmod-ed
    (scratch / "mytg.session").write_bytes(b"")  # synthetic placeholder
    os.chmod(scratch / "mytg.session", 0o644)
    assert tgc.has_session(scratch / "mytg")
    tgc.secure_session_file(scratch / "mytg")
    assert (scratch / "mytg.session").stat().st_mode & 0o777 == 0o600
    assert tgc.lock_path(scratch / "mytg", allow_string=False) == scratch / "mytg.session.lock"


def test_session_lock_keeps_sink_and_pull_apart(monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
                                                capsys: pytest.CaptureFixture[str]) -> None:
    import subprocess
    import sys
    import time

    from gvon import telegram_client as tgc

    monkeypatch.setenv("TG_SESSION", "")
    monkeypatch.setenv("TG_API_ID", "12345")
    monkeypatch.setenv("TG_API_HASH", "synthetic-hash-value")
    sess = tmp_path / "telegram.session"
    sess.write_bytes(b"")  # synthetic placeholder file, not a real session
    # another process (standing in for the always-on sink) holds the lock
    holder = subprocess.Popen(
        [sys.executable, "-c",
         "import sys, time; from gvon import telegram_client as t; "
         f"l = t.acquire_session_lock('sink (make telegram)', {str(sess)!r}, allow_string=False); "
         "print('held', flush=True); time.sleep(30)"],
        stdout=subprocess.PIPE, text=True)
    try:
        assert holder.stdout.readline().strip() == "held"
        assert tgc.session_in_use(sess).startswith("sink (make telegram) pid=")
        assert tgc.main(["ready", "--session", str(sess)]) == 3
        err = capsys.readouterr().err
        assert "in use by sink" in err and "synthetic-hash-value" not in err
        with pytest.raises(tgc.SessionBusyError):
            tgc.acquire_session_lock("telegram_ingest pull", sess)
        # pull refuses before touching Telethon (exit 2, the ConfigError path), never "database is locked"
        assert ti.main(["--session", str(sess), "pull"]) == 2
        assert "in use by another gvon process" in capsys.readouterr().err
    finally:
        holder.kill()
        holder.wait()
    deadline = time.time() + 5
    while tgc.session_in_use(sess) and time.time() < deadline:
        time.sleep(0.05)
    assert tgc.session_in_use(sess) is None and tgc.main(["ready", "--session", str(sess)]) == 0
    with tgc.acquire_session_lock("telegram_ingest pull", sess):
        with pytest.raises(tgc.SessionBusyError):  # a second holder in the same process is refused too
            tgc.acquire_session_lock("sink (make telegram)", sess)
    assert tgc.session_in_use(sess) is None
