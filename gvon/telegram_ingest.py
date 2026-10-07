"""Ingest a week of Telegram engagement (user account, via Telethon) into the same raw shape as X tweets.

Why this shape:
- The X pipeline (label -> train -> nullify) reads one JSONL row per post with fixed keys. Telegram rows
  use the SAME keys where they mean something (id, text, author_*, created_at, conversation_id,
  in_reply_to_user_id, kind, referenced, ...) plus platform/chat_title/chat_type, so labeling and training
  need no Telegram-specific join logic.
- A user account (not a bot) is the only way to read your own DMs and groups. Login is interactive
  (phone + code, optional 2FA password) and is done ONCE by the user with `login`; afterwards the session
  lives in data/telegram.session (gitignored, chmod 600) or, alternatively, in a TG_SESSION StringSession.
- What is collected per chat, within --since-days:
    private chats  every incoming message ("dm"); your own messages are kept as "own" context.
    groups/supergroups
                   if you posted in the window: every message (cap --max-scan-per-chat scanned);
                   replies to your messages => "group_reply", messages that mention you => "group_mention",
                   the rest => "group", your own => "own".
                   if you did not post: only messages that mention you (Telegram's "my mentions" search,
                   which also covers replies to your older messages) => "group_mention"/"group_reply".
    broadcast channels you own/admin: their linked discussion group ("comments") is scanned like a group
                   you posted in, and replies to the channel's posts count as replies to you.
    other broadcast channels, Saved Messages and Telegram's service account (777000, login codes) are
    never collected.
  Each chat keeps at most --max-per-chat rows (replies/mentions/DMs first, then own, then plain group
  messages); how many were dropped, by kind, is logged and written to the summary.
- The message each collected row replies to is fetched (batched) and written to telegram_context.jsonl,
  so the teacher sees "replying to @x: <text>" exactly like on X.
- Resumable: data/raw/_telegram_state.json stores, per chat, the highest message id already examined;
  the next run only asks for newer messages (min_id) and merges them into the existing files (deduped by
  id, rows older than the window dropped). Output and state are flushed together every few chats and on
  any error, so a crash or Ctrl-C loses at most the chats since the last flush.
- FloodWaitError: Telethon sleeps through short waits itself (flood_sleep_threshold, 60s); a longer one
  is caught here, logged, and slept for the requested seconds capped at 15 minutes, then the chat is
  retried (up to 3 times).
- One inaccessible chat does not abort the run: any other Telegram RPC error (e.g. ChannelPrivateError for a
  supergroup you were banned from) is logged with the chat id and error type only, counted in the summary's
  "chats_failed" by error name, and the chat is skipped (its state is not advanced, so the next run retries
  it). ChannelForbidden / ChatForbidden dialogs are skipped up front ("forbidden" in chats_skipped).
- `pull` and `login` hold the session's exclusive lock (gvon.telegram_client.SessionLock) while connected:
  they refuse to run while the sink (`make telegram`) uses the same session, instead of failing inside
  Telethon with "database is locked".
- Message text, names and chat titles are written only under data/ (gitignored); logs carry chat ids,
  types and counts only. API id/hash/session strings are never printed.

The Telethon client is wrapped by TelethonAPI; tests inject a fake with the same five methods, so
everything below it (classification, row shapes, caps, resume, flood handling) runs without a network.

CLI:
    python -m gvon.telegram_ingest login
    python -m gvon.telegram_ingest pull [--since-days 7] [--chats N] [--max-per-chat 300]
                                        [--max-scan-per-chat 3000] [--skip-bots] [--fresh] [-v]
"""
from __future__ import annotations

import argparse
import asyncio
import getpass
import json
import logging
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Awaitable, Callable, Iterable, Protocol

from gvon import env
from gvon import telegram_client as tgc

log = logging.getLogger("gvon.telegram_ingest")

DEFAULT_OUT = env.ROOT / "data" / "raw"  # repo-anchored: real messages never land outside gitignored data/
DEFAULT_SESSION = tgc.DEFAULT_SESSION  # data/telegram.session (GVON_TG_SESSION overrides, see gvon.telegram_client)
TWEETS_NAME = "telegram.jsonl"
USERS_NAME = "telegram_users.jsonl"
CONTEXT_NAME = "telegram_context.jsonl"
STATE_NAME = "_telegram_state.json"
SUMMARY_NAME = "_telegram_summary.json"
STATE_VERSION = 1

DEFAULT_SINCE_DAYS = 7.0
DEFAULT_MAX_PER_CHAT = 300
DEFAULT_MAX_SCAN = 3000
FLOOD_WAIT_CAP_S = 15 * 60
MAX_FLOOD_RETRIES = 3
TELETHON_FLOOD_SLEEP_S = 60  # waits up to this are slept inside Telethon; longer ones raise to us
GET_MESSAGES_BATCH = 100
FLUSH_EVERY_CHATS = 10
SERVICE_USER_IDS = tgc.SERVICE_USER_IDS  # Telegram's own notifications (login codes): never collected
FORBIDDEN_ENTITY_TYPES = ("ChannelForbidden", "ChatForbidden")  # banned/kicked: history is not readable

KINDS = ("dm", "group_reply", "group_mention", "group", "own")
# Lower keeps first when a chat is over --max-per-chat.
KIND_PRIORITY = {"dm": 0, "group_reply": 0, "group_mention": 1, "own": 2, "group": 3}
PRIVATE_TYPES = ("private", "bot")


# --------------------------------------------------------------------------- pure helpers


def iso(dt: datetime | None) -> str | None:
    """Telegram dates are aware UTC datetimes; naive ones are taken as UTC."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def as_utc(dt: datetime | None) -> datetime | None:
    if dt is None:
        return None
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt


def msg_key(chat_id: int | str, message_id: int | str) -> str:
    """Row id: f"{chat_id}:{msg_id}" (message ids are only unique within a chat)."""
    return f"{chat_id}:{message_id}"


def chat_type(entity: Any) -> str:
    """private | bot | group | supergroup | channel, from the Telethon entity's attributes.

    User has `bot`; Channel has `broadcast`/`megagroup`/`gigagroup`; a basic Chat has none of them."""
    if hasattr(entity, "bot"):
        return "bot" if getattr(entity, "bot", False) else "private"
    if getattr(entity, "broadcast", False):
        return "channel"
    if getattr(entity, "megagroup", False) or getattr(entity, "gigagroup", False):
        return "supergroup"
    return "group"


def display_name(entity: Any) -> str:
    if entity is None:
        return ""
    title = getattr(entity, "title", None)
    if title:
        return str(title)
    parts = [getattr(entity, "first_name", None), getattr(entity, "last_name", None)]
    return " ".join(str(p) for p in parts if p)


def username_of(entity: Any) -> str | None:
    name = getattr(entity, "username", None) if entity is not None else None
    if not name:  # Telegram users can carry several collectible usernames; the first active one is theirs
        for u in getattr(entity, "usernames", None) or []:
            if getattr(u, "active", True) and getattr(u, "username", None):
                return str(u.username)
        return None
    return str(name)


def reply_target(msg: Any) -> int | None:
    """Id of the message this one replies to, in the same chat; None for non-replies.

    Forum-topic messages carry the topic root as reply_to_msg_id with no reply_to_top_id: that is the
    topic, not a reply. Replies to another chat's message (reply_to_peer_id) cannot be resolved here."""
    rt = getattr(msg, "reply_to", None)
    if rt is None:
        return None
    mid = getattr(rt, "reply_to_msg_id", None)
    if mid is None or getattr(rt, "reply_to_peer_id", None) is not None:
        return None
    if getattr(rt, "forum_topic", False) and getattr(rt, "reply_to_top_id", None) is None:
        return None
    return int(mid)


def is_service(msg: Any) -> bool:
    """Join/leave/pin/... service messages carry an `action` and no user text."""
    return getattr(msg, "action", None) is not None


def media_kind(msg: Any) -> str | None:
    """'photo', 'document', 'webpage', ... from the media class name (MessageMediaPhoto -> photo)."""
    media = getattr(msg, "media", None)
    if media is None:
        return None
    name = type(media).__name__
    if name.startswith("MessageMedia"):
        name = name[len("MessageMedia"):]
    return name.lower() or "media"


def message_text(msg: Any) -> str:
    """The raw message text (no markdown re-rendering)."""
    return str(getattr(msg, "message", None) or "")


def is_own(msg: Any, own_sender_ids: set[int]) -> bool:
    return bool(getattr(msg, "out", False)) or getattr(msg, "sender_id", None) in own_sender_ids


def classify(msg: Any, *, ctype: str, own_sender_ids: set[int], own_msg_ids: set[int], posted: bool,
             reply_sender_id: int | None = None) -> str | None:
    """Kind of a message for the label stage, or None when it is not collected.

    own > dm (private/bot chats) > group_reply (replies to your message, or to your channel's post) >
    group_mention (Telegram's `mentioned` flag) > group (only in chats where you posted in the window)."""
    if is_own(msg, own_sender_ids):
        return "own"
    if ctype in PRIVATE_TYPES:
        return "dm"
    target = reply_target(msg)
    if target is not None and (target in own_msg_ids or (reply_sender_id is not None and reply_sender_id in own_sender_ids)):
        return "group_reply"
    if getattr(msg, "mentioned", False):
        return "group_mention"
    if posted:
        return "group"
    return None


def message_row(msg: Any, *, chat_id: int, chat_title: str, ctype: str, kind: str, sender: Any,
                sender_id: int | None, reply_sender_id: int | None) -> dict[str, Any]:
    """One telegram.jsonl / telegram_context.jsonl row (tweets.jsonl keys + platform/chat_title/chat_type/media)."""
    target = reply_target(msg)
    return {
        "id": msg_key(chat_id, msg.id),
        "text": message_text(msg),
        "author_id": str(sender_id) if sender_id is not None else None,
        "author_username": username_of(sender),
        "author_name": display_name(sender),
        "created_at": iso(getattr(msg, "date", None)),
        "conversation_id": str(chat_id),
        "in_reply_to_user_id": str(reply_sender_id) if reply_sender_id is not None else None,
        "lang": None,
        "kind": kind,
        "referenced": [{"type": "replied_to", "id": msg_key(chat_id, target)}] if target is not None else [],
        "public_metrics": {},
        "platform": "telegram",
        "chat_title": chat_title,
        "chat_type": ctype,
        "media": media_kind(msg),
    }


def user_row(user_id: int | str, entity: Any, *, is_self: bool = False) -> dict[str, Any]:
    """telegram_users.jsonl row. Bios need one GetFullUser call per user (flood-prone), so description is null."""
    return {
        "id": str(user_id),
        "username": username_of(entity),
        "name": display_name(entity),
        "description": None,
        "platform": "telegram",
        "bot": bool(getattr(entity, "bot", False)),
        "scam": bool(getattr(entity, "scam", False)),
        "fake": bool(getattr(entity, "fake", False)),
        "verified": bool(getattr(entity, "verified", False)),
        "premium": bool(getattr(entity, "premium", False)),
        "is_self": is_self,
    }


def apply_cap(candidates: list[tuple[Any, str]], max_per_chat: int) -> tuple[list[tuple[Any, str]], Counter[str]]:
    """Keep at most max_per_chat (msg, kind) pairs: by KIND_PRIORITY, newest first. Returns (kept, dropped by kind)."""
    ordered = sorted(candidates, key=lambda mk: (KIND_PRIORITY[mk[1]], -int(mk[0].id)))
    kept = ordered[:max(0, max_per_chat)]
    dropped = Counter(kind for _, kind in ordered[max(0, max_per_chat):])
    kept.sort(key=lambda mk: int(mk[0].id))
    return kept, dropped


def flood_wait_seconds(exc: BaseException) -> int | None:
    """Seconds a FloodWaitError (or FloodPremiumWait / SlowModeWait) asks us to wait; None for other errors."""
    secs = getattr(exc, "seconds", None)
    if isinstance(secs, int) and not isinstance(secs, bool) and "Wait" in type(exc).__name__:
        return secs
    return None


# --------------------------------------------------------------------------- state + files


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{lineno}: bad JSON: {e}") from e
    return rows


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    atomic_write(path, "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def new_state(account_id: int | None) -> dict[str, Any]:
    return {"version": STATE_VERSION, "account_id": account_id, "chats": {}}


def load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return new_state(None)
    try:
        st = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        log.warning("%s is corrupt; starting from scratch", path.name)
        return new_state(None)
    if not isinstance(st, dict) or st.get("version") != STATE_VERSION or not isinstance(st.get("chats"), dict):
        return new_state(None)
    return st


def last_seen_id(state: dict[str, Any], chat_id: int) -> int:
    entry = state["chats"].get(str(chat_id)) or {}
    try:
        return int(entry.get("last_id") or 0)
    except (TypeError, ValueError):
        return 0


@dataclass
class Store:
    """Merged rows on disk + this run's additions; flushed together with the state."""
    out_dir: Path
    since: datetime
    rows: dict[str, dict[str, Any]] = field(default_factory=dict)
    context: dict[str, dict[str, Any]] = field(default_factory=dict)
    users: dict[str, dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def load(cls, out_dir: Path, since: datetime, fresh: bool) -> "Store":
        s = cls(out_dir=out_dir, since=since)
        if not fresh:
            s.rows = {r["id"]: r for r in load_jsonl(out_dir / TWEETS_NAME) if "id" in r}
            s.context = {r["id"]: r for r in load_jsonl(out_dir / CONTEXT_NAME) if "id" in r}
            s.users = {str(u["id"]): u for u in load_jsonl(out_dir / USERS_NAME) if "id" in u}
        return s

    def add(self, rows: list[dict[str, Any]], context: list[dict[str, Any]], users: dict[str, dict[str, Any]]) -> None:
        for r in rows:
            self.rows[r["id"]] = r
            self.context.pop(r["id"], None)
        for c in context:
            if c["id"] not in self.rows:
                self.context[c["id"]] = c
        for uid, u in users.items():
            if uid in self.users and self.users[uid].get("is_self") and not u.get("is_self"):
                continue
            self.users[uid] = u

    def flush(self, state: dict[str, Any], state_path: Path) -> None:
        cutoff = iso(self.since) or ""
        rows = sorted((r for r in self.rows.values() if (r.get("created_at") or "") >= cutoff),
                      key=lambda r: (r.get("created_at") or "", r["id"]))
        refs = {ref["id"] for r in rows for ref in r.get("referenced") or []}
        kept_ids = {r["id"] for r in rows}
        context = sorted((c for cid, c in self.context.items() if cid in refs and cid not in kept_ids),
                         key=lambda r: (r.get("created_at") or "", r["id"]))
        authors = {r.get("author_id") for r in rows} | {c.get("author_id") for c in context}
        users = sorted((u for uid, u in self.users.items() if uid in authors or u.get("is_self")), key=lambda u: u["id"])
        write_jsonl(self.out_dir / TWEETS_NAME, rows)
        write_jsonl(self.out_dir / CONTEXT_NAME, context)
        write_jsonl(self.out_dir / USERS_NAME, users)
        atomic_write(state_path, json.dumps(state, indent=2) + "\n")


# --------------------------------------------------------------------------- client seam


class TelegramAPI(Protocol):
    """The five calls the collector makes. TelethonAPI implements them; tests pass a fake."""

    async def me(self) -> Any: ...
    def dialogs(self) -> AsyncIterator[Any]: ...
    def messages(self, entity: Any, *, min_id: int = 0, from_me: bool = False, mentions: bool = False) -> AsyncIterator[Any]: ...
    async def get_messages(self, entity: Any, ids: list[int]) -> list[Any]: ...
    async def linked_chat(self, entity: Any) -> tuple[int, Any] | None: ...


class TelethonAPI:
    """Thin adapter over a connected, authorized telethon.TelegramClient (newest-first iteration)."""

    def __init__(self, client: Any) -> None:
        self.client = client

    async def me(self) -> Any:
        return await self.client.get_me()

    def dialogs(self) -> AsyncIterator[Any]:
        return self.client.iter_dialogs()

    def messages(self, entity: Any, *, min_id: int = 0, from_me: bool = False, mentions: bool = False) -> AsyncIterator[Any]:
        kw: dict[str, Any] = {"min_id": min_id}
        if from_me:
            kw["from_user"] = "me"
        if mentions:
            from telethon.tl.types import InputMessagesFilterMyMentions

            kw["filter"] = InputMessagesFilterMyMentions
        return self.client.iter_messages(entity, **kw)

    async def get_messages(self, entity: Any, ids: list[int]) -> list[Any]:
        got = await self.client.get_messages(entity, ids=ids)
        return list(got or [])

    async def linked_chat(self, entity: Any) -> tuple[int, Any] | None:
        """(peer id, entity) of a broadcast channel's discussion group, or None."""
        from telethon import utils
        from telethon.tl.functions.channels import GetFullChannelRequest

        full = await self.client(GetFullChannelRequest(entity))
        linked = getattr(full.full_chat, "linked_chat_id", None)
        if not linked:
            return None
        for chat in full.chats:
            if chat.id == linked:
                return utils.get_peer_id(chat), chat
        return None


# --------------------------------------------------------------------------- collection


@dataclass
class ChatResult:
    rows: list[dict[str, Any]]
    context: list[dict[str, Any]]
    users: dict[str, dict[str, Any]]
    last_id: int
    scanned: int
    truncated: bool
    mode: str
    dropped: Counter[str]


@dataclass
class ChatJob:
    chat_id: int
    entity: Any
    title: str
    ctype: str
    top_id: int = 0
    force_scan: bool = False  # discussion group of a channel you own: treat as "you posted"
    extra_own: set[int] = field(default_factory=set)  # e.g. the owning channel's peer id


async def window(it: AsyncIterator[Any], since: datetime, max_scan: int) -> tuple[list[Any], int, bool]:
    """Newest-first messages down to `since` (service messages skipped). Returns (msgs, max id seen, truncated)."""
    out: list[Any] = []
    top = 0
    n = 0
    async for m in it:
        top = max(top, int(m.id))
        d = as_utc(getattr(m, "date", None))
        if d is not None and d < since:
            break
        n += 1
        if n > max_scan:
            return out, top, True
        if not is_service(m):
            out.append(m)
    return out, top, False


def sender_entity(msg: Any, job: ChatJob, me: Any) -> Any:
    s = getattr(msg, "sender", None)
    if s is not None:
        return s
    if job.ctype in PRIVATE_TYPES:  # in a DM the peer is the only other party
        return me if getattr(msg, "out", False) else job.entity
    return None


def sender_id_of(msg: Any, job: ChatJob, me_id: int) -> int | None:
    sid = getattr(msg, "sender_id", None)
    if sid is not None:
        return int(sid)
    if getattr(msg, "out", False):
        return me_id
    if job.ctype in PRIVATE_TYPES:
        return job.chat_id
    return None


async def collect_chat(api: TelegramAPI, job: ChatJob, *, me: Any, me_id: int, own_sender_ids: set[int],
                       since: datetime, min_id: int, max_per_chat: int, max_scan: int) -> ChatResult:
    own_ids = set(own_sender_ids) | job.extra_own
    truncated = False
    if job.ctype in PRIVATE_TYPES:
        msgs, top, truncated = await window(api.messages(job.entity, min_id=min_id), since, max_scan)
        mode, posted = "dm", False
    else:
        own_msgs, top_own, _ = await window(api.messages(job.entity, min_id=min_id, from_me=True), since, max_scan)
        posted = job.force_scan or bool(own_msgs)
        if posted:
            msgs, top, truncated = await window(api.messages(job.entity, min_id=min_id), since, max_scan)
            mode = "scan"
        else:
            try:
                msgs, top, truncated = await window(api.messages(job.entity, min_id=min_id, mentions=True), since, max_scan)
                mode = "mentions"
            except Exception as e:  # the mentions search is not available everywhere: fall back to a scan
                if flood_wait_seconds(e) is not None:
                    raise
                log.info("chat %s: mentions search failed (%s); scanning instead", job.chat_id, type(e).__name__)
                msgs, top, truncated = await window(api.messages(job.entity, min_id=min_id), since, max_scan)
                msgs = [m for m in msgs if getattr(m, "mentioned", False)]
                mode = "mentions_scan"
            top = max(top, job.top_id)  # nothing newer than the dialog's top message can exist
        top = max(top, top_own)
    by_id = {int(m.id): m for m in msgs}
    own_msg_ids = {mid for mid, m in by_id.items() if is_own(m, own_ids)}

    candidates: list[tuple[Any, str]] = []
    for m in msgs:
        kind = classify(m, ctype=job.ctype, own_sender_ids=own_ids, own_msg_ids=own_msg_ids, posted=posted)
        if kind is not None:
            candidates.append((m, kind))
    kept, dropped = apply_cap(candidates, max_per_chat)
    if dropped:
        log.info("chat %s (%s): kept %d of %d rows (cap --max-per-chat %d); dropped %s", job.chat_id, job.ctype,
                 len(kept), len(candidates), max_per_chat, dict(dropped))

    # Fetch the replied-to messages we do not have, for context and to spot replies to your older posts.
    wanted = sorted({t for m, _ in kept if (t := reply_target(m)) is not None and t not in by_id})
    fetched: dict[int, Any] = {}
    for i in range(0, len(wanted), GET_MESSAGES_BATCH):
        for m in await api.get_messages(job.entity, wanted[i:i + GET_MESSAGES_BATCH]):
            if m is not None and not is_service(m):
                fetched[int(m.id)] = m
    lookup = {**fetched, **by_id}

    rows: list[dict[str, Any]] = []
    context: list[dict[str, Any]] = []
    users: dict[str, dict[str, Any]] = {}
    kept_ids = {int(m.id) for m, _ in kept}

    def note_user(sid: int | None, entity: Any) -> None:
        if sid is not None and entity is not None and str(sid) not in users:
            users[str(sid)] = user_row(sid, entity, is_self=(sid == me_id))

    for m, kind in kept:
        target = reply_target(m)
        parent = lookup.get(target) if target is not None else None
        parent_sender = sender_id_of(parent, job, me_id) if parent is not None else None
        if kind in ("group_mention", "group") and parent is not None and (parent_sender in own_ids or is_own(parent, own_ids)):
            kind = "group_reply"
        sid = sender_id_of(m, job, me_id)
        sender = sender_entity(m, job, me)
        rows.append(message_row(m, chat_id=job.chat_id, chat_title=job.title, ctype=job.ctype, kind=kind, sender=sender,
                                sender_id=sid, reply_sender_id=parent_sender))
        note_user(sid, sender)
        if parent is not None and target not in kept_ids:
            psender = sender_entity(parent, job, me)
            context.append(message_row(parent, chat_id=job.chat_id, chat_title=job.title, ctype=job.ctype, kind="context",
                                       sender=psender, sender_id=parent_sender, reply_sender_id=None))
            note_user(parent_sender, psender)
    return ChatResult(rows=rows, context=context, users=users, last_id=max(top, min_id), scanned=len(msgs),
                      truncated=truncated, mode=mode, dropped=dropped)


async def with_flood_retry(make: Callable[[], Awaitable[Any]], what: str,
                           sleep: Callable[[float], Awaitable[Any]] = asyncio.sleep) -> Any:
    """Run make(); on a flood wait sleep min(requested, 15 min) and retry, at most MAX_FLOOD_RETRIES times."""
    for attempt in range(MAX_FLOOD_RETRIES + 1):
        try:
            return await make()
        except Exception as e:
            secs = flood_wait_seconds(e)
            if secs is None or attempt >= MAX_FLOOD_RETRIES:
                raise
            wait = min(max(int(secs), 1), FLOOD_WAIT_CAP_S)
            log.warning("flood wait: Telegram asked for %ds during %s; sleeping %ds (cap %ds), retry %d/%d",
                        secs, what, wait, FLOOD_WAIT_CAP_S, attempt + 1, MAX_FLOOD_RETRIES)
            await sleep(wait)
    raise AssertionError("unreachable")


def is_rpc_error(exc: BaseException) -> bool:
    """A Telegram RPC error (telethon.errors.RPCError), i.e. Telegram refused this one request."""
    try:
        from telethon.errors import RPCError
    except ImportError:  # pragma: no cover
        return False
    return isinstance(exc, RPCError)


async def list_dialogs(api: TelegramAPI) -> list[Any]:
    return [d async for d in api.dialogs()]


def can_post_as(entity: Any) -> bool:
    """You own or admin this broadcast channel, so its posts (and the comments under them) are yours."""
    return bool(getattr(entity, "creator", False) or getattr(entity, "admin_rights", None))


async def plan_jobs(api: TelegramAPI, dialogs: list[Any], *, since: datetime, skip_bots: bool,
                    sleep: Callable[[float], Awaitable[Any]]) -> tuple[list[ChatJob], set[int], Counter[str]]:
    """Turn dialogs into chat jobs (most recent first). Returns (jobs, owned channel ids, skipped by reason)."""
    jobs: dict[int, ChatJob] = {}
    order: list[int] = []
    skipped: Counter[str] = Counter()
    owned_channels: set[int] = set()
    linked: list[tuple[int, int, Any]] = []  # (channel id, linked chat id, linked entity)
    for d in dialogs:
        entity = d.entity
        chat_id = int(d.id)
        ctype = chat_type(entity)
        if getattr(entity, "is_self", False):
            skipped["saved_messages"] += 1
            continue
        if chat_id in SERVICE_USER_IDS:
            skipped["telegram_service"] += 1
            continue
        if type(entity).__name__ in FORBIDDEN_ENTITY_TYPES:
            skipped["forbidden"] += 1
            continue
        if ctype == "bot" and skip_bots:
            skipped["bot"] += 1
            continue
        if ctype == "channel":
            if can_post_as(entity):
                owned_channels.add(chat_id)
                got = await with_flood_retry(lambda e=entity: api.linked_chat(e), f"linked chat of {chat_id}", sleep)
                if got is not None:
                    linked.append((chat_id, int(got[0]), got[1]))
            else:
                skipped["broadcast_channel"] += 1
            continue
        d_date = as_utc(getattr(d, "date", None))
        if d_date is not None and d_date < since:
            skipped["inactive_in_window"] += 1
            continue
        top = getattr(getattr(d, "message", None), "id", 0) or 0
        jobs[chat_id] = ChatJob(chat_id=chat_id, entity=entity, title=getattr(d, "name", None) or display_name(entity),
                                ctype=ctype, top_id=int(top))
        order.append(chat_id)
    for channel_id, lid, lentity in linked:
        job = jobs.get(lid)
        if job is None:
            job = jobs[lid] = ChatJob(chat_id=lid, entity=lentity, title=display_name(lentity), ctype=chat_type(lentity))
            order.append(lid)
        job.force_scan = True
        job.extra_own.add(channel_id)
    return [jobs[c] for c in order], owned_channels, skipped


async def pull(api: TelegramAPI, *, out_dir: Path = DEFAULT_OUT, since_days: float = DEFAULT_SINCE_DAYS,
               max_chats: int | None = None, max_per_chat: int = DEFAULT_MAX_PER_CHAT,
               max_scan: int = DEFAULT_MAX_SCAN, skip_bots: bool = False, fresh: bool = False,
               now: datetime | None = None, sleep: Callable[[float], Awaitable[Any]] = asyncio.sleep,
               flush_every: int = FLUSH_EVERY_CHATS) -> dict[str, Any]:
    """Collect every in-scope chat into out_dir/telegram*.jsonl; returns the run summary (also written)."""
    now = now or datetime.now(timezone.utc)
    since = now - timedelta(days=since_days)
    me = await with_flood_retry(api.me, "get_me", sleep)
    me_id = int(me.id)
    state_path = out_dir / STATE_NAME
    state = load_state(state_path)
    if fresh or state.get("account_id") not in (None, me_id):
        if state.get("account_id") not in (None, me_id):
            log.info("state belongs to a different Telegram account; starting fresh")
            fresh = True
        state = new_state(me_id)
    state["account_id"] = me_id
    store = Store.load(out_dir, since, fresh)
    store.users[str(me_id)] = user_row(me_id, me, is_self=True)

    dialogs = await with_flood_retry(lambda: list_dialogs(api), "dialog listing", sleep)
    jobs, owned_channels, skipped = await plan_jobs(api, dialogs, since=since, skip_bots=skip_bots, sleep=sleep)
    if max_chats is not None:
        jobs = jobs[:max(0, max_chats)]
    own_sender_ids = {me_id} | owned_channels
    log.info("dialogs=%d chats_to_pull=%d skipped=%s window_start=%s", len(dialogs), len(jobs), dict(skipped), iso(since))

    totals: Counter[str] = Counter()
    dropped: Counter[str] = Counter()
    modes: Counter[str] = Counter()
    failed: Counter[str] = Counter()
    truncated_chats = 0
    pending = 0
    try:
        for i, job in enumerate(jobs, 1):
            min_id = last_seen_id(state, job.chat_id)
            try:
                res: ChatResult = await with_flood_retry(
                    lambda j=job, mi=min_id: collect_chat(api, j, me=me, me_id=me_id, own_sender_ids=own_sender_ids,
                                                          since=since, min_id=mi, max_per_chat=max_per_chat,
                                                          max_scan=max_scan),
                    f"chat {job.chat_id}", sleep)
            except Exception as e:  # noqa: BLE001 - isolated per chat below; flood waits still abort the run
                if flood_wait_seconds(e) is not None or not is_rpc_error(e):
                    raise
                failed[type(e).__name__] += 1
                log.warning("chat %s (%s) skipped: %s", job.chat_id, job.ctype, type(e).__name__)
                continue
            store.add(res.rows, res.context, res.users)
            state["chats"][str(job.chat_id)] = {"last_id": res.last_id, "type": job.ctype,
                                                "updated_at": iso(datetime.now(timezone.utc))}
            totals.update(r["kind"] for r in res.rows)
            dropped.update(res.dropped)
            modes[res.mode] += 1
            truncated_chats += int(res.truncated)
            if res.truncated:
                log.info("chat %s (%s): scan stopped at --max-scan-per-chat %d before the window start",
                         job.chat_id, job.ctype, max_scan)
            log.debug("chat %d/%d id=%s type=%s mode=%s rows=%d", i, len(jobs), job.chat_id, job.ctype, res.mode, len(res.rows))
            pending += 1
            if pending >= flush_every:
                store.flush(state, state_path)
                pending = 0
    finally:
        store.flush(state, state_path)

    rows_now = [r for r in store.rows.values() if (r.get("created_at") or "") >= (iso(since) or "")]
    summary = {
        "generated_at": iso(now), "window_start": iso(since), "since_days": since_days,
        "dialogs": len(dialogs), "chats_pulled": len(jobs) - sum(failed.values()), "chats_skipped": dict(skipped),
        "chats_failed": dict(failed),
        "chat_modes": dict(modes), "rows_this_run": dict(totals), "dropped_by_cap_this_run": dict(dropped),
        "chats_scan_truncated": truncated_chats, "max_per_chat": max_per_chat, "max_scan_per_chat": max_scan,
        "rows_total": len(rows_now), "rows_total_by_kind": dict(Counter(r["kind"] for r in rows_now)),
        "context_rows": len(store.context), "users": len(store.users),
    }
    atomic_write(out_dir / SUMMARY_NAME, json.dumps(summary, indent=2) + "\n")
    return summary


# --------------------------------------------------------------------------- client construction + CLI


ConfigError = tgc.TelegramConfigError  # missing/invalid TG_API_ID / TG_API_HASH; messages never carry values


def api_credentials() -> tuple[int, str]:
    """TG_API_ID / TG_API_HASH from the environment or .env (values never echoed)."""
    return tgc.load_creds()


def restrict_session_file(session_path: Path) -> None:
    tgc.secure_session_file(session_path)


async def login(client: Any, session_path: Path) -> int:
    """Interactive phone + code (+ 2FA password) login. Prints nothing except the prompts."""
    from telethon.errors import SessionPasswordNeededError

    await client.connect()
    try:
        if not await client.is_user_authorized():
            phone = input("Telegram phone number (international format, e.g. +15551234567): ").strip()
            sent = await client.send_code_request(phone)
            code = input("Login code (sent by Telegram): ").strip()
            try:
                await client.sign_in(phone=phone, code=code, phone_code_hash=sent.phone_code_hash)
            except SessionPasswordNeededError:
                await client.sign_in(password=getpass.getpass("Two-step verification password: "))
    finally:
        await client.disconnect()
    restrict_session_file(session_path)
    return 0


async def run_pull(args: argparse.Namespace, session_path: Path) -> int:
    with tgc.acquire_session_lock("telegram_ingest pull", session_path):  # SessionBusyError if the sink runs
        return await _run_pull(args, session_path)


async def _run_pull(args: argparse.Namespace, session_path: Path) -> int:
    client = tgc.make_client(session_path, flood_sleep_threshold=TELETHON_FLOOD_SLEEP_S)  # TG_SESSION wins if set
    await client.connect()
    try:
        if not await client.is_user_authorized():
            print("not logged in to Telegram: run `python -m gvon.telegram_ingest login` once (interactive)", file=sys.stderr)
            return 2
        try:
            summary = await pull(TelethonAPI(client), out_dir=Path(args.out), since_days=args.since_days,
                                 max_chats=args.chats, max_per_chat=args.max_per_chat,
                                 max_scan=args.max_scan_per_chat, skip_bots=args.skip_bots, fresh=args.fresh)
        except Exception as e:  # finished chats are already flushed with their state: a rerun resumes
            print(f"FAILED: {type(e).__name__}: {str(e)[:300]}. Chats finished before the error are saved; "
                  "re-run `pull` to continue.", file=sys.stderr)
            return 1
    finally:
        await client.disconnect()
    restrict_session_file(session_path)
    print(json.dumps(summary, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m gvon.telegram_ingest", description=__doc__.splitlines()[0])
    ap.add_argument("--session", default=None,
                    help="Telethon SQLite session file (default: GVON_TG_SESSION, else data/telegram.session; "
                         "pull uses TG_SESSION instead when that StringSession is set)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("login", help="interactive one-time login (phone + code); writes the session file")
    p = sub.add_parser("pull", help="collect the last --since-days of Telegram engagement into data/raw/telegram*.jsonl")
    p.add_argument("--since-days", type=float, default=DEFAULT_SINCE_DAYS)
    p.add_argument("--chats", type=int, default=None, help="pull at most N chats (most recently active first)")
    p.add_argument("--max-per-chat", type=int, default=DEFAULT_MAX_PER_CHAT)
    p.add_argument("--max-scan-per-chat", type=int, default=DEFAULT_MAX_SCAN,
                   help="stop scanning a chat after this many messages in the window")
    p.add_argument("--skip-bots", action="store_true", help="skip private chats with bots")
    p.add_argument("--fresh", action="store_true", help="ignore saved state and existing rows; re-pull the window")
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    session_path = tgc.normalize_session(args.session) if args.session else tgc.session_path()
    try:
        if args.cmd == "login":
            logging.basicConfig(level=logging.ERROR)
            with tgc.acquire_session_lock("telegram_ingest login", session_path, allow_string=False):
                client = tgc.make_client(session_path, allow_string=False,  # login always writes the session file
                                         flood_sleep_threshold=TELETHON_FLOOD_SLEEP_S)
                return asyncio.run(login(client, session_path))
        logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                            format="%(asctime)s %(levelname)s %(message)s")
        logging.getLogger("telethon").setLevel(logging.INFO if args.verbose else logging.WARNING)
        return asyncio.run(run_pull(args, session_path))
    except ConfigError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
