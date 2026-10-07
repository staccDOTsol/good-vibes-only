"""Telegram SINK: a client running as you that removes nullified incoming messages from your view.

Why a client and not a proxy: Telegram speaks MTProto with its own encryption, so the mitmproxy trick used
for x.com cannot see or edit it. The closest thing to "it never reaches my eyes" is a user-session client
(Telethon) that reacts to every incoming message within milliseconds and deletes / mutes / archives it
before you open the chat. Official clients still RECEIVE the message (and may notify) until the sink acts;
see docs/TELEGRAM.md for limits and the notification advice.

Run:   make telegram                      (= .venv/bin/python -m gvon.telegram_nullifier)
       python -m gvon.telegram_nullifier --backfill 50     also score the last 50 unread messages per dialog
       python -m gvon.telegram_nullifier --vault 20        print the last 20 vault entries (audit) and exit
Login: once, interactively, `make telegram-login` (writes data/telegram.session). The sink never prompts.

Decision per incoming text message: blocklist hit (sender id or username) or the local classifier
(gvon.classifier.Nullifier.should_nullify: score >= threshold, watch-list / author-aggregate rules).
A student trained with author inputs (config student_inputs has author_bio / author_stats) also gets the
sender's author context: {verified} from the sender entity and, for author_bio models, the bio (`about`)
from one GetFullUserRequest. Both are fetched lazily and cached per sender id (AUTHOR_CACHE_TTL_S), never
per message, and only when the text would actually be scored. Telegram exposes no follower counts or
account creation date, so those stay unknown, exactly as in the Telegram training rows. The bio lookup is
bounded (BIO_TIMEOUT_S) and skipped while Telegram has GetFullUser flood-limited, so a live decision never
waits on it. The classifier runs in a worker thread, never on the event loop, and score() runs only for
messages that are nullified (it is needed only for the vault).

Never acted on: Telegram's service account 777000 (login codes, new-login alerts), Telegram support accounts
(`support` flag), your own messages, and senders listed in GVON_TG_ALLOW.

Environment:
  GVON_TG_DM_ACTION     delete_for_me | mute | archive | log    (default delete_for_me)  private chats
  GVON_TG_GROUP_ACTION  delete_for_me | mute | archive | log    (default log)            groups / channels
  GVON_TG_GROUP_DELETE  1 => in groups (not broadcast channels) where you are an admin with delete rights,
                        delete for everyone (never auto-forwarded channel posts or posts sent as a channel)
  GVON_TG_MARK_READ     1 => after a fallback mute in a group, mark the chat read, but only when the flagged
                        message is the chat's only unread message, and never during --backfill (off by default:
                        marking read sends read receipts)
  GVON_TG_ALLOW         comma-separated sender ids / usernames that are never acted on
  GVON_TG_MEDIA_MAX_MB  largest attachment saved to the vault before a delete (default 50); a message whose
                        attachment cannot be saved is logged instead of deleted
  GVON_DRY_RUN          1 => decide, log and write the vault, but make no Telegram calls
  GVON_MODEL_DIR / GVON_BLOCKLIST / GVON_THRESHOLD   as for the X proxy
  TG_SESSION            a Telethon StringSession: used by the sink and `pull` (never by `login`) INSTEAD of the
                        session file; the startup log names the account it authorizes as
  GVON_TG_SESSION       override data/telegram.session

Actions:
  delete_for_me   messages.deleteMessages(revoke=False): gone from your history on all your devices. No read
                  receipt is sent. Only possible in private chats and basic groups; in supergroups/channels it
                  falls back to mute (the message stays visible). In a basic group a permanent permission
                  error also falls back to mute; in a DM, and for any other error, the chat is left alone and
                  the vault records "error: <name>".
  mute            notify settings mute_until = far future (no more notifications from that chat).
  archive         mute + move the chat to the Archive folder (folder 1). Telegram un-archives an UNMUTED
                  chat when a new message arrives, so archive always mutes first.
  log             vault entry only.
  delete_for_everyone   (GVON_TG_GROUP_DELETE=1 and you are admin) messages deleted with revoke=True.
A message whose chat cannot be resolved is never acted on (result "skipped: unresolved chat").

Vault: every nullified message (text, entities, reply/forward headers, attachment metadata) is appended to
data/telegram_nullified.jsonl (mode 0600) BEFORE any action runs. Before a delete, a photo/document is
downloaded to data/telegram_media/<chat>/<msg>/ (files 0600) and its path recorded; if that fails or the file
is over GVON_TG_MEDIA_MAX_MB, the message is logged instead of deleted. Terminal logs never include message text.
"""
from __future__ import annotations

import argparse
import asyncio
import functools
import inspect
import json
import logging
import os
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable

from gvon import env
from gvon import telegram_client as tgc

try:  # telethon is a runtime dependency, but decide()/apply() must import and test without a session
    from telethon.errors import (AuthKeyError, ChatAdminRequiredError, FloodWaitError, ForbiddenError, RPCError,
                                 UnauthorizedError)
except ImportError:  # pragma: no cover
    class RPCError(Exception):  # type: ignore[no-redef]
        pass

    class FloodWaitError(RPCError):  # type: ignore[no-redef]
        seconds = 0

    class ForbiddenError(RPCError):  # type: ignore[no-redef]
        pass

    class ChatAdminRequiredError(RPCError):  # type: ignore[no-redef]
        pass

    class UnauthorizedError(RPCError):  # type: ignore[no-redef]
        pass

    class AuthKeyError(RPCError):  # type: ignore[no-redef]
        pass

# delete_for_me in a basic group falls back to mute only on these (the delete can never succeed there)
PERMANENT_DELETE_ERRORS = (ForbiddenError, ChatAdminRequiredError)
# the session itself is dead (revoked, logged out, key unregistered/duplicated): reconnecting cannot help
AUTH_ERRORS = (UnauthorizedError, AuthKeyError)

log = logging.getLogger("gvon.telegram_nullifier")

ACTIONS = ("delete_for_me", "mute", "archive", "log")
DELETE_FOR_EVERYONE = "delete_for_everyone"
DEFAULT_DM_ACTION = "delete_for_me"
DEFAULT_GROUP_ACTION = "log"
PLATFORM = "telegram"
VAULT_PATH = env.ROOT / "data" / "telegram_nullified.jsonl"
MUTE_FOREVER = 2**31 - 1  # max int32 unix time: Telegram's "muted forever"
ARCHIVE_FOLDER = 1
FLOOD_RETRIES = 3
ADMIN_CACHE_S = 600.0
AUTHOR_CACHE_TTL_S = 86_400.0
AUTHOR_CACHE_MAX = 5_000
BIO_TIMEOUT_S = 1.0       # a live decision never waits longer than this for a sender's bio
BIO_RETRY_S = 60.0        # after a bio timeout / flood wait, retry that sender's bio after this long
MEDIA_DIR_NAME = "telegram_media"
DEFAULT_MEDIA_MAX_MB = 50.0
MEDIA_TIMEOUT_S = 120.0
RECONNECT_MIN_S = 5.0
RECONNECT_MAX_S = 300.0
RECONNECT_STABLE_S = 120.0  # the backoff resets only after a connection has stayed up this long
EXEMPT_SENDER_IDS = tgc.SERVICE_USER_IDS  # Telegram's service account (login codes, security alerts)

Sleep = Callable[[float], Awaitable[Any]]


def _truthy(v: str | None) -> bool:
    return (v or "").strip().lower() in ("1", "true", "yes", "on")


def _action_from_env(key: str, default: str) -> str:
    v = (env.get(key) or "").strip().lower() or default
    if v not in ACTIONS:
        raise ValueError(f"{key}={v!r} is not one of {', '.join(ACTIONS)}")
    return v


def _allow_from_env() -> frozenset[str]:
    """GVON_TG_ALLOW: comma/space separated sender ids or usernames (with or without @), lower-cased."""
    raw = (env.get("GVON_TG_ALLOW") or "").replace(",", " ").split()
    return frozenset(x.lstrip("@").lower() for x in raw if x.lstrip("@"))


def _float_from_env(key: str, default: float) -> float:
    raw = (env.get(key) or "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        raise ValueError(f"{key}={raw!r} is not a number") from None


@dataclass(frozen=True)
class TgConfig:
    dm_action: str = DEFAULT_DM_ACTION
    group_action: str = DEFAULT_GROUP_ACTION
    group_delete: bool = False
    dry_run: bool = False
    mark_read: bool = False
    allow: frozenset[str] = frozenset()
    media_max_mb: float = DEFAULT_MEDIA_MAX_MB

    def __post_init__(self) -> None:
        for name in ("dm_action", "group_action"):
            if getattr(self, name) not in ACTIONS:
                raise ValueError(f"{name}={getattr(self, name)!r} is not one of {', '.join(ACTIONS)}")

    @classmethod
    def from_env(cls, *, dry_run: bool | None = None) -> "TgConfig":
        return cls(
            dm_action=_action_from_env("GVON_TG_DM_ACTION", DEFAULT_DM_ACTION),
            group_action=_action_from_env("GVON_TG_GROUP_ACTION", DEFAULT_GROUP_ACTION),
            group_delete=_truthy(env.get("GVON_TG_GROUP_DELETE")),
            dry_run=_truthy(env.get("GVON_DRY_RUN")) if dry_run is None else bool(dry_run),
            mark_read=_truthy(env.get("GVON_TG_MARK_READ")),
            allow=_allow_from_env(),
            media_max_mb=_float_from_env("GVON_TG_MEDIA_MAX_MB", DEFAULT_MEDIA_MAX_MB),
        )


# ----------------------------------------------------------------------------------------------- decision

def _accepts(fn: Any, name: str) -> bool:
    """True when callable `fn` declares a parameter called `name` (explicitly, not just **kwargs)."""
    try:
        return name in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


def _identity_kwargs(fn: Any, sender_id: Any, username: str | None) -> dict[str, Any]:
    kw: dict[str, Any] = {"author_id": None if sender_id is None else str(sender_id), "username": username}
    if _accepts(fn, "platform"):
        kw["platform"] = PLATFORM
    return kw


def wants_author(nullifier: Any) -> bool:
    """Does this decider take author context (a v2 student whose should_nullify declares `author`)?"""
    return bool(getattr(nullifier, "uses_author", False)) and _accepts(getattr(nullifier, "should_nullify", None), "author")


def evaluate(text: str, sender_id: Any, username: str | None, is_private: bool, is_admin: bool,
             nullifier: Any, cfg: TgConfig, author: dict[str, Any] | None = None
             ) -> tuple[bool, float | None, str | None, str | None]:
    """decide() plus the reason: (should_nullify, score, action, reason).

    author: optional sender context {bio, followers, following, tweet_count, created_at, verified}, passed
    only to deciders that declare it. reason is one of blocklist | score | watchlist | author_aggregate
    (None when not nullified). score (for the vault) is computed only for nullified messages, and not at
    all for a blocklist hit while text scoring is off; it is None when not computed.
    Synchronous and CPU-bound (a model forward pass): the sink runs it in a worker thread."""
    blocked = False
    is_blocked = getattr(nullifier, "is_blocked", None)
    if callable(is_blocked):
        blocked = bool(is_blocked(**_identity_kwargs(is_blocked, sender_id, username)))
    kw = _identity_kwargs(nullifier.should_nullify, sender_id, username)
    if author is not None and _accepts(nullifier.should_nullify, "author"):
        kw["author"] = author
    should = bool(nullifier.should_nullify(text, **kw))
    if not should:
        return False, None, None, None
    score: float | None = None
    if not blocked or getattr(nullifier, "text_model_enabled", True):
        try:  # cached inside Nullifier, so this usually does not re-encode after should_nullify()
            score_fn = nullifier.score
            score = float((score_fn([text], authors=[author]) if author is not None and _accepts(score_fn, "authors")
                           else score_fn([text]))[0])
        except Exception:  # noqa: BLE001 - a scoring failure must not hide a blocklist decision
            log.exception("gvon-tg: score() failed")
    if blocked:
        reason = "blocklist"
    else:
        thr = getattr(nullifier, "threshold", None)
        is_watched = getattr(nullifier, "is_watched", None)
        if thr is not None and score is not None and score >= float(thr):
            reason = "score"
        elif callable(is_watched) and is_watched(**_identity_kwargs(is_watched, sender_id, username)):
            reason = "watchlist"
        else:
            reason = "author_aggregate"
    if is_private:
        action = cfg.dm_action
    elif is_admin and cfg.group_delete:
        action = DELETE_FOR_EVERYONE
    else:
        action = cfg.group_action
    return True, score, action, reason


def decide(text: str, sender_id: Any, username: str | None, is_private: bool, is_admin: bool,
           nullifier: Any, cfg: TgConfig, author: dict[str, Any] | None = None
           ) -> tuple[bool, float | None, str | None]:
    """Pure decision: (should_nullify, score, action). action and score are None when the message is kept."""
    should, score, action, _ = evaluate(text, sender_id, username, is_private, is_admin, nullifier, cfg, author)
    return should, score, action


# ------------------------------------------------------------------------------------------------ actions

async def _flood_safe(make_call: Callable[[], Awaitable[Any]], sleep: Sleep = asyncio.sleep) -> Any:
    """Run a Telegram call; on FloodWaitError sleep the requested time and retry (up to FLOOD_RETRIES)."""
    for attempt in range(FLOOD_RETRIES + 1):
        try:
            return await make_call()
        except FloodWaitError as e:
            if attempt >= FLOOD_RETRIES:
                raise
            wait = int(getattr(e, "seconds", 0) or 0) + 1
            log.warning("gvon-tg: FloodWait %ss (attempt %d/%d), sleeping", wait, attempt + 1, FLOOD_RETRIES)
            await sleep(wait)
    return None  # pragma: no cover


async def _mute(client: Any, chat: Any, sleep: Sleep) -> None:
    from telethon.tl.functions.account import UpdateNotifySettingsRequest
    from telethon.tl.types import InputNotifyPeer, InputPeerNotifySettings

    peer = await _flood_safe(lambda: client.get_input_entity(chat), sleep)
    req = UpdateNotifySettingsRequest(peer=InputNotifyPeer(peer=peer),
                                      settings=InputPeerNotifySettings(mute_until=MUTE_FOREVER))
    await _flood_safe(lambda: client(req), sleep)


async def _mark_read_if_sole_unread(client: Any, chat: Any, ids: list[int], sleep: Sleep) -> bool:
    """Mark the chat read up to max(ids), but ONLY when the chat has no unread messages besides the flagged
    ones, so no legitimate unread message is marked read. Marking read sends read receipts (DMs, small
    groups); callers opt in (GVON_TG_MARK_READ=1) and never call this during backfill."""
    from telethon.tl.functions.messages import GetPeerDialogsRequest
    from telethon.tl.types import InputDialogPeer

    try:
        peer = await _flood_safe(lambda: client.get_input_entity(chat), sleep)
        res = await _flood_safe(lambda: client(GetPeerDialogsRequest(peers=[InputDialogPeer(peer=peer)])), sleep)
        dialogs = list(getattr(res, "dialogs", None) or [])
        if not dialogs or int(getattr(dialogs[0], "unread_count", 0) or 0) > len(ids):
            return False
        await _flood_safe(lambda: client.send_read_acknowledge(chat, max_id=max(ids)), sleep)
        return True
    except Exception as e:  # noqa: BLE001 - the mute already happened; never fail the action on this
        log.info("gvon-tg: mark-read skipped (%s)", type(e).__name__)
        return False


async def apply(action: str, client: Any, chat: Any, msg_ids: int | list[int], *, dry_run: bool = False,
                is_channel: bool = False, is_private: bool = False, chat_key: Any = None,
                state: dict[str, set] | None = None, mark_read: bool = False,
                sleep: Sleep = asyncio.sleep) -> str:
    """Carry out `action` with any Telethon-like client. Returns a short result string for the vault/log.

    is_channel: the chat is a supergroup or broadcast channel (delete-for-me impossible there).
    is_private: a DM; a failed delete there is recorded as an error, never turned into a mute.
    state: optional {"muted": set(), "archived": set()} so a chat is muted/archived once per process.
    mark_read: after a FALLBACK mute only, mark the chat read when the flagged messages are its only unread
    ones. Never used after a successful delete_for_me (the message is already gone, and a read
    acknowledgement would send read receipts and mark earlier unread messages read)."""
    ids = [msg_ids] if isinstance(msg_ids, int) else list(msg_ids)
    if action not in ACTIONS + (DELETE_FOR_EVERYONE,):
        raise ValueError(f"unknown action {action!r}")
    if dry_run:
        return "dry_run"
    if action == "log":
        return "logged"
    key = chat_key if chat_key is not None else getattr(chat, "id", chat)
    muted = state.setdefault("muted", set()) if state is not None else set()
    archived = state.setdefault("archived", set()) if state is not None else set()

    async def mute_once() -> None:
        if key not in muted:
            await _mute(client, chat, sleep)
            muted.add(key)

    async def fallback_mute(why: str) -> str:
        await mute_once()
        marked = mark_read and await _mark_read_if_sole_unread(client, chat, ids, sleep)
        return f"fallback_mute ({why}){' + marked_read' if marked else ''}"

    if action == DELETE_FOR_EVERYONE:
        await _flood_safe(lambda: client.delete_messages(chat, ids, revoke=True), sleep)
        return "deleted_for_everyone"
    if action == "delete_for_me":
        if is_channel:
            return await fallback_mute("delete_for_me impossible in supergroups/channels")
        try:
            await _flood_safe(lambda: client.delete_messages(chat, ids, revoke=False), sleep)
        except FloodWaitError:
            raise
        except PERMANENT_DELETE_ERRORS as e:  # e.g. MessageDeleteForbiddenError in a basic group
            if is_private:  # never mute a whole DM over one message: the caller records the error
                raise
            log.warning("gvon-tg: delete_for_me refused (%s); falling back to mute", type(e).__name__)
            return await fallback_mute(type(e).__name__)
        return "deleted_for_me"
    if action == "mute":
        await mute_once()
        return "muted"
    # archive
    await mute_once()
    if key not in archived:
        await _flood_safe(lambda: client.edit_folder(chat, ARCHIVE_FOLDER), sleep)
        archived.add(key)
    return "archived"


# -------------------------------------------------------------------------------------------------- vault

def vault_append(entry: dict[str, Any], path: Path = VAULT_PATH) -> None:
    """Append one JSON line to the vault (created 0600 under a 0700 directory)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(entry, ensure_ascii=False, default=str) + "\n"
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
    try:
        os.write(fd, line.encode("utf-8"))
    finally:
        os.close(fd)


def vault_entries(path: Path = VAULT_PATH) -> list[dict[str, Any]]:
    """All vault entries (oldest first) with their later {"update": "result"} lines folded in.
    Corrupt lines are skipped."""
    path = Path(path)
    if not path.exists():
        return []
    entries: list[dict[str, Any]] = []
    index: dict[tuple[Any, Any], dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(e, dict):
            continue
        key = (e.get("chat_id"), e.get("msg_id"))
        if e.get("update") == "result":
            if key in index:
                index[key]["result"] = e.get("result")
            continue
        entries.append(e)
        index[key] = e
    return entries


def vault_tail(n: int, path: Path = VAULT_PATH) -> list[dict[str, Any]]:
    """Last n vault entries (oldest first), results folded in."""
    return vault_entries(path)[-n:] if n > 0 else []


def format_vault_entry(e: dict[str, Any]) -> str:
    who = e.get("sender_username") and f"@{e['sender_username']}" or str(e.get("sender_id"))
    score = e.get("score")
    score_s = f"{score:.3f}" if isinstance(score, (int, float)) else "-"
    head = (f"{e.get('ts', '?')}  {e.get('action', '?')}->{e.get('result', '?')}  score={score_s}  "
            f"reason={e.get('reason')}  chat={e.get('chat_title') or ''} ({e.get('chat_id')})  from={who}  "
            f"msg={e.get('msg_id')}{'  [backfill]' if e.get('backfill') else ''}")
    text = str(e.get("text", "")).replace("\n", "\n    ")
    media = e.get("media")
    if isinstance(media, dict):
        where = media.get("path") or f"not saved: {media.get('error') or 'no file'}"
        text += f"\n    [{media.get('type')}: {where}]"
    return f"{head}\n    {text}"


# --------------------------------------------------------------------------------------------------- sink

def _display_name(entity: Any) -> str | None:
    if entity is None:
        return None
    title = getattr(entity, "title", None)
    if title:
        return str(title)
    parts = [getattr(entity, "first_name", None), getattr(entity, "last_name", None)]
    name = " ".join(p for p in parts if p)
    return name or getattr(entity, "username", None)


def _peer_is_channel(peer: Any) -> bool:
    try:
        from telethon.tl.types import InputPeerChannel, PeerChannel

        return isinstance(peer, (PeerChannel, InputPeerChannel))
    except ImportError:  # pragma: no cover
        return False


def _revocable_chat(chat: Any) -> bool:
    """A resolved group entity that is NOT a broadcast channel: the only place delete-for-everyone may run."""
    if chat is None or type(chat).__name__.startswith("InputPeer"):
        return False
    return not getattr(chat, "broadcast", False)


def _is_channel_post(msg: Any, sender: Any) -> bool:
    """A post sent as a channel (auto-forwarded into its discussion group, or a channel / anonymous admin
    posting as the chat): deleting it for everyone would remove a channel post or break its comment thread."""
    fwd = getattr(msg, "fwd_from", None)
    if fwd is not None and getattr(fwd, "saved_from_peer", None) is not None:
        return True
    if _peer_is_channel(getattr(msg, "from_id", None)):
        return True
    return sender is not None and (type(sender).__name__ in ("Channel", "ChannelForbidden")
                                   or bool(getattr(sender, "broadcast", False) or getattr(sender, "megagroup", False)))


def _to_plain(obj: Any) -> Any:
    """Telethon TLObject -> dict for the vault (None stays None)."""
    if obj is None:
        return None
    to_dict = getattr(obj, "to_dict", None)
    if callable(to_dict):
        try:
            return to_dict()
        except Exception:  # noqa: BLE001
            pass
    return str(obj)


DOWNLOADABLE_MEDIA = ("photo", "document")


def media_info(msg: Any) -> dict[str, Any] | None:
    """Vault metadata for the message's attachment, or None. type is 'photo', 'document', 'webpage', ...;
    photos/documents get file metadata (mime_type, size, name, file_id) and later a local path."""
    media = getattr(msg, "media", None)
    if media is None:
        return None
    name = type(media).__name__
    kind = (name[len("MessageMedia"):] if name.startswith("MessageMedia") else name).lower() or "media"
    info: dict[str, Any] = {"type": kind, "path": None}
    if kind in DOWNLOADABLE_MEDIA:
        f = getattr(msg, "file", None)
        for key in ("mime_type", "size", "name"):
            try:
                info[key] = getattr(f, key, None)
            except Exception:  # noqa: BLE001
                info[key] = None
        try:
            fid = getattr(f, "id", None)
            info["file_id"] = fid if isinstance(fid, str) else None
        except Exception:  # noqa: BLE001
            info["file_id"] = None
    elif kind != "webpage":  # link previews are rebuilt from the URL in the text; keep other media whole
        info["detail"] = _to_plain(media)
    return info


def _is_channel(chat: Any) -> bool:
    """Supergroup / gigagroup / broadcast channel (Telegram 'Channel' type): no delete-for-me there."""
    try:
        from telethon.tl.types import Channel, ChannelForbidden

        if isinstance(chat, (Channel, ChannelForbidden)):
            return True
    except ImportError:  # pragma: no cover
        pass
    return bool(getattr(chat, "megagroup", False) or getattr(chat, "broadcast", False)
                or getattr(chat, "gigagroup", False))


class TelegramSink:
    """Glues decide()/apply()/vault to a client. Works with any Telethon-like client (tests use fakes)."""

    def __init__(self, client: Any, nullifier: Any, cfg: TgConfig, *, vault_path: Path = VAULT_PATH,
                 media_dir: Path | None = None, sleep: Sleep = asyncio.sleep) -> None:
        self.client = client
        self.nullifier = nullifier
        self.cfg = cfg
        self.vault_path = Path(vault_path)
        self.media_dir = Path(media_dir) if media_dir is not None else self.vault_path.parent / MEDIA_DIR_NAME
        self.sleep = sleep
        self.state: dict[str, set] = {"muted": set(), "archived": set()}
        self._admin: dict[Any, tuple[float, bool]] = {}
        self._entities: OrderedDict[Any, tuple[float, Any]] = OrderedDict()  # key -> (expires_at, value)
        self._author_ctx: OrderedDict[Any, tuple[float, dict[str, Any]]] = OrderedDict()
        self.stats = {"seen": 0, "nullified": 0, "errors": 0, "entity_lookups": 0, "bio_lookups": 0,
                      "exempt": 0}

    @staticmethod
    def _lru_get(cache: OrderedDict, key: Any) -> Any:
        hit = cache.get(key)
        if hit is None or time.monotonic() >= hit[0]:
            return None
        cache.move_to_end(key)
        return hit[1]

    @staticmethod
    def _lru_put(cache: OrderedDict, key: Any, value: Any, ttl: float = AUTHOR_CACHE_TTL_S) -> None:
        cache[key] = (time.monotonic() + ttl, value)
        cache.move_to_end(key)
        while len(cache) > AUTHOR_CACHE_MAX:
            cache.popitem(last=False)

    async def sender_entity(self, sender_id: Any, sender: Any) -> Any:
        """The sender entity: the one the event carried, else one get_entity call per sender id (cached,
        failures included, so an unresolvable id is not retried on every message)."""
        if sender is not None or sender_id is None:
            return sender
        cached = self._lru_get(self._entities, sender_id)
        if cached is not None:
            return cached or None
        entity: Any = None
        try:
            self.stats["entity_lookups"] += 1
            entity = await self.client.get_entity(sender_id)
        except Exception as e:  # noqa: BLE001 - unknown/hidden senders are normal
            log.info("gvon-tg: get_entity failed for sender %s (%s)", sender_id, type(e).__name__)
        self._lru_put(self._entities, sender_id, entity if entity is not None else False)
        return entity

    def _needs_author(self, text: str, sender_id: Any, username: str | None) -> bool:
        """Author context matters only when the text will actually be scored by an author-aware student."""
        if not wants_author(self.nullifier) or not getattr(self.nullifier, "text_model_enabled", True):
            return False
        try:
            from gvon.classifier import is_low_info

            if is_low_info(text):
                return False
        except ImportError:  # pragma: no cover
            pass
        is_blocked = getattr(self.nullifier, "is_blocked", None)
        return not (callable(is_blocked) and is_blocked(**_identity_kwargs(is_blocked, sender_id, username)))

    async def author_context(self, sender_id: Any, sender: Any) -> dict[str, Any] | None:
        """{bio, followers, following, tweet_count, created_at, verified} for this sender, cached per sender id.

        verified comes from the entity; the bio (`about`) needs GetFullUserRequest and is fetched only for
        author_bio students and only for users (not channels posting as themselves). Counts and creation date
        do not exist on Telegram and stay None."""
        if sender_id is None:
            return None
        cached = self._lru_get(self._author_ctx, sender_id)
        if cached is not None:
            return cached
        verified = getattr(sender, "verified", None) if sender is not None else None
        ctx: dict[str, Any] = {"bio": None, "followers": None, "following": None, "tweet_count": None,
                               "created_at": None, "verified": verified if isinstance(verified, bool) else None}
        is_user = sender is not None and hasattr(sender, "first_name") and not hasattr(sender, "title")
        ttl = AUTHOR_CACHE_TTL_S
        if getattr(self.nullifier, "uses_author_bio", False) and is_user:
            from telethon.tl.functions.users import GetFullUserRequest

            # Telethon sleeps through flood waits <= flood_sleep_threshold (60 s) INSIDE the call, and also
            # pre-sleeps later calls of a flood-limited request type. Never let a live decision wait on
            # that: skip while the request type is flood-limited, and bound the call itself.
            if self._flood_limited(GetFullUserRequest.CONSTRUCTOR_ID):
                log.info("gvon-tg: GetFullUser flood-limited; scoring sender %s without bio", sender_id)
                ttl = BIO_RETRY_S
            else:
                try:
                    self.stats["bio_lookups"] += 1
                    full = await asyncio.wait_for(self.client(GetFullUserRequest(sender)), timeout=BIO_TIMEOUT_S)
                    about = getattr(getattr(full, "full_user", None), "about", None)
                    ctx["bio"] = about if isinstance(about, str) else None
                except FloodWaitError as e:  # score without the bio; retry this sender's bio later
                    log.warning("gvon-tg: GetFullUser FloodWait %ss; scoring sender %s without bio",
                                e.seconds, sender_id)
                    ttl = BIO_RETRY_S
                except asyncio.TimeoutError:
                    log.info("gvon-tg: GetFullUser took > %.1fs; scoring sender %s without bio", BIO_TIMEOUT_S,
                             sender_id)
                    ttl = BIO_RETRY_S
                except Exception as e:  # noqa: BLE001
                    log.info("gvon-tg: GetFullUser failed for sender %s (%s)", sender_id, type(e).__name__)
        self._lru_put(self._author_ctx, sender_id, ctx, ttl)
        return ctx

    def _flood_limited(self, constructor_id: int) -> bool:
        """True while Telethon has this request type in a flood wait (it would sleep before sending)."""
        waited = getattr(self.client, "_flood_waited_requests", None)
        due = waited.get(constructor_id) if isinstance(waited, dict) else None
        return isinstance(due, (int, float)) and due - time.time() > 3

    def _exempt(self, sender_id: Any, sender: Any = None, username: str | None = None) -> bool:
        """Never act on Telegram's service account (777000), Telegram support accounts, or GVON_TG_ALLOW."""
        try:
            if sender_id is not None and int(sender_id) in EXEMPT_SENDER_IDS:
                return True
        except (TypeError, ValueError):
            pass
        if sender is not None and getattr(sender, "support", False) is True:
            return True
        allow = self.cfg.allow
        return bool(allow) and ((sender_id is not None and str(sender_id) in allow)
                                or (bool(username) and str(username).lower() in allow))

    async def _save_media(self, msg: Any, chat_id: Any, msg_id: int, info: dict[str, Any]) -> bool:
        """Download a photo/document into media_dir/<chat>/<msg>/ (dirs 0700, file 0600) before a destructive
        action. Fills info["path"] (or info["error"]) and returns whether the file is safely on disk."""
        size = info.get("size")
        limit = int(self.cfg.media_max_mb * 1024 * 1024)
        if isinstance(size, int) and size > limit:
            info["error"] = f"larger than GVON_TG_MEDIA_MAX_MB={self.cfg.media_max_mb:g}"
            return False
        target = self.media_dir / str(chat_id) / str(msg_id)
        try:
            for d in (self.media_dir, self.media_dir / str(chat_id), target):
                tgc.ensure_private_dir(d)
            got = await asyncio.wait_for(self.client.download_media(msg, file=str(target)), timeout=MEDIA_TIMEOUT_S)
        except Exception as e:  # noqa: BLE001 - includes asyncio.TimeoutError
            info["error"] = type(e).__name__
            return False
        if not got or not Path(str(got)).is_file():
            info["error"] = "download returned no file"
            return False
        path = Path(str(got))
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass
        try:
            info["path"] = str(path.resolve().relative_to(env.ROOT.resolve()))
        except ValueError:
            info["path"] = str(path)
        return True

    async def is_admin(self, chat: Any) -> bool:
        """Admin with delete rights in this group? Only asked when GVON_TG_GROUP_DELETE=1; cached 10 min."""
        key = getattr(chat, "id", chat)
        hit = self._admin.get(key)
        now = time.monotonic()
        if hit and now - hit[0] < ADMIN_CACHE_S:
            return hit[1]
        try:
            perms = await _flood_safe(lambda: self.client.get_permissions(chat, "me"), self.sleep)
            ok = bool(perms and getattr(perms, "is_admin", False) and getattr(perms, "delete_messages", True))
        except Exception as e:  # noqa: BLE001
            log.warning("gvon-tg: could not read admin rights for chat %s (%s); assuming not admin",
                        key, type(e).__name__)
            ok = False
        self._admin[key] = (now, ok)
        return ok

    async def handle_message(self, msg: Any, chat: Any, sender: Any, *, is_private: bool,
                             backfill: bool = False) -> dict[str, Any] | None:
        """Decide and act on one message. Returns the vault entry when nullified, else None. Never raises.

        chat may be None (Telethon could not resolve it): the message is still decided and vaulted, but no
        Telegram action runs (result "skipped: unresolved chat"), because a None entity would make Telethon
        delete by id in the private/basic-group id space, i.e. some unrelated message."""
        t0 = time.perf_counter()
        try:
            if getattr(msg, "out", False):
                return None
            text = getattr(msg, "message", None) or getattr(msg, "raw_text", None) or ""
            if not str(text).strip():
                return None
            self.stats["seen"] += 1
            sender_id = getattr(msg, "sender_id", None)
            if self._exempt(sender_id):
                self.stats["exempt"] += 1
                return None
            sender = await self.sender_entity(sender_id, sender)
            username = getattr(sender, "username", None)
            if self._exempt(sender_id, sender, username):
                self.stats["exempt"] += 1
                return None
            author = (await self.author_context(sender_id, sender)
                      if self._needs_author(str(text), sender_id, username) else None)
            is_channel = _is_channel(chat) or _peer_is_channel(getattr(msg, "peer_id", None))
            admin = ((not is_private) and self.cfg.group_delete and _revocable_chat(chat)
                     and not _is_channel_post(msg, sender) and await self.is_admin(chat))
            loop = asyncio.get_running_loop()  # model forward pass off the event loop (Telethon keeps running)
            should, score, action, reason = await loop.run_in_executor(
                None, functools.partial(evaluate, str(text), sender_id, username, is_private, admin,
                                        self.nullifier, self.cfg, author))
            if not should:
                return None
            chat_id = getattr(chat, "id", None)
            if chat_id is None:
                chat_id = getattr(msg, "chat_id", None)
            msg_id = int(getattr(msg, "id"))
            date = getattr(msg, "date", None)
            reply_to = getattr(msg, "reply_to", None)
            media = media_info(msg)
            entry: dict[str, Any] = {
                "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "msg_date": date.isoformat() if hasattr(date, "isoformat") else date,
                "chat_id": chat_id, "chat_title": _display_name(chat), "is_private": bool(is_private),
                "msg_id": msg_id, "sender_id": sender_id, "sender_username": username, "text": str(text),
                "entities": [_to_plain(e) for e in (getattr(msg, "entities", None) or [])] or None,
                "reply_to_msg_id": getattr(reply_to, "reply_to_msg_id", None),
                "fwd_from": _to_plain(getattr(msg, "fwd_from", None)), "media": media,
                "score": None if score is None else round(float(score), 4), "reason": reason, "action": action,
                "dry_run": self.cfg.dry_run, "backfill": backfill, "result": "pending",
            }
            destructive = action == DELETE_FOR_EVERYONE or (action == "delete_for_me" and not is_channel)
            if self.cfg.dry_run or action == "log":  # no Telegram call: the result is known up front
                entry["result"] = await apply(action, self.client, chat, msg_id, dry_run=self.cfg.dry_run)
                vault_append(entry, self.vault_path)
            elif chat is None:
                entry["result"] = "skipped: unresolved chat"
                vault_append(entry, self.vault_path)
            elif (destructive and media is not None and media["type"] in DOWNLOADABLE_MEDIA
                  and not await self._save_media(msg, chat_id, msg_id, media)):
                # the attachment could not be kept: never destroy what the vault cannot hold
                entry["result"] = f"logged (media not saved: {media.get('error')})"
                vault_append(entry, self.vault_path)
            else:
                # vault first: even if the action fails or the process dies, the message is kept locally;
                # the outcome follows as a small {"update": "result"} line (folded in by vault_entries)
                vault_append(entry, self.vault_path)
                try:
                    entry["result"] = await apply(action, self.client, chat, msg_id, is_channel=is_channel,
                                                  is_private=is_private, chat_key=chat_id, state=self.state,
                                                  mark_read=self.cfg.mark_read and not backfill, sleep=self.sleep)
                except Exception as e:  # noqa: BLE001
                    entry["result"] = f"error: {type(e).__name__}"
                    self.stats["errors"] += 1
                    log.exception("gvon-tg: action %s failed for chat=%s msg=%s", action, chat_id, msg_id)
                vault_append({"ts": entry["ts"], "chat_id": chat_id, "msg_id": msg_id, "update": "result",
                              "result": entry["result"]}, self.vault_path)
            self.stats["nullified"] += 1
            log.info("gvon-tg: nullified chat=%s msg=%s score=%s reason=%s action=%s result=%s in %.0fms%s",
                     chat_id, msg_id, "-" if score is None else f"{score:.3f}", reason, action, entry["result"],
                     (time.perf_counter() - t0) * 1000, " [backfill]" if backfill else "")
            return entry
        except Exception:  # noqa: BLE001 - never crash the update loop
            self.stats["errors"] += 1
            log.exception("gvon-tg: failed to handle message")
            return None

    async def on_new_message(self, event: Any) -> None:
        """events.NewMessage(incoming=True) handler."""
        try:
            chat = await event.get_chat()
            if chat is None:  # unresolvable entity: fall back to the input peer (never a bare None target)
                get_input = getattr(event, "get_input_chat", None)
                chat = await get_input() if callable(get_input) else None
            sender = await event.get_sender()
            await self.handle_message(event.message, chat, sender, is_private=bool(event.is_private))
        except Exception:  # noqa: BLE001
            self.stats["errors"] += 1
            log.exception("gvon-tg: event handler failed")

    async def backfill(self, n: int) -> int:
        """Score the last n unread incoming messages of every dialog with unread messages. Returns #nullified."""
        if n <= 0:
            return 0
        hits = 0
        async for dialog in self.client.iter_dialogs():
            unread = int(getattr(dialog, "unread_count", 0) or 0)
            if unread <= 0:
                continue
            try:
                async for msg in self.client.iter_messages(dialog.entity, limit=min(n, unread)):
                    if getattr(msg, "out", False):
                        continue
                    sender = await msg.get_sender() if hasattr(msg, "get_sender") else None
                    if await self.handle_message(msg, dialog.entity, sender, is_private=bool(dialog.is_user),
                                                 backfill=True):
                        hits += 1
            except FloodWaitError as e:
                log.warning("gvon-tg: backfill FloodWait %ss on dialog %s; sleeping, skipping rest of it",
                            e.seconds, getattr(dialog, "id", "?"))
                await self.sleep(int(e.seconds) + 1)
            except Exception:  # noqa: BLE001
                log.exception("gvon-tg: backfill failed for dialog %s", getattr(dialog, "id", "?"))
        log.info("gvon-tg: backfill done: %d nullified", hits)
        return hits


# ---------------------------------------------------------------------------------------------------- run

def build_nullifier() -> Any:
    """gvon.classifier.Nullifier, built once (logs device + load ms). Without a trained model, falls back
    LOUDLY to blocklist-only decisions, like the X proxy."""
    model_dir = Path(env.get("GVON_MODEL_DIR") or env.ROOT / "models" / "latest")
    blocklist = Path(env.get("GVON_BLOCKLIST") or env.ROOT / "data" / "blocklist.json")
    t0 = time.perf_counter()
    try:
        from gvon.classifier import Nullifier

        nul = Nullifier(model_dir, blocklist)
        warm = getattr(nul, "warmup", None)
        if callable(warm):
            warm()
        log.info("gvon-tg: Nullifier loaded on device=%s threshold=%.3f (source=%s, text_model=%s) in %.0fms",
                 getattr(nul, "device", "?"), float(getattr(nul, "threshold", float("nan"))),
                 getattr(nul, "threshold_source", "?"), getattr(nul, "text_model_enabled", "?"),
                 (time.perf_counter() - t0) * 1000)
        return nul
    except Exception as exc:  # noqa: BLE001
        from gvon.prune import BlocklistDecider

        log.warning("gvon-tg: DEGRADED MODE, classifier unavailable (%s: %s); blocklist %s only (%.0fms)",
                    type(exc).__name__, exc, blocklist, (time.perf_counter() - t0) * 1000)
        return BlocklistDecider(blocklist)


def _auth_lost(e: BaseException) -> tgc.TelegramConfigError:
    return tgc.TelegramConfigError(
        f"the Telegram session is no longer authorized ({type(e).__name__}): it was terminated in Settings -> "
        "Devices, logged out, or used from elsewhere. Run `make telegram-login` again")


async def reconnect_loop(client: Any, *, sleep: Sleep = asyncio.sleep,
                         clock: Callable[[], float] = time.monotonic) -> None:
    """Run until disconnected, then reconnect with exponential backoff (5 s .. 5 min). The backoff resets only
    after a connection stayed up RECONNECT_STABLE_S, not right after connect(): a revoked session still
    connects (help.getConfig works unauthenticated) and fails moments later. A permanent authorization error
    (UnauthorizedError, AuthKeyError incl. AuthKeyDuplicated) raises TelegramConfigError instead of looping."""
    delay = RECONNECT_MIN_S
    while True:
        started = clock()
        try:
            await client.run_until_disconnected()
        except AUTH_ERRORS as e:
            raise _auth_lost(e) from None
        except Exception:  # noqa: BLE001
            log.exception("gvon-tg: connection loop error")
        if clock() - started >= RECONNECT_STABLE_S:
            delay = RECONNECT_MIN_S
        log.warning("gvon-tg: disconnected; reconnecting in %.0fs", delay)
        await sleep(delay)
        delay = min(delay * 2, RECONNECT_MAX_S)
        try:
            await client.connect()
        except AUTH_ERRORS as e:
            raise _auth_lost(e) from None
        except Exception:  # noqa: BLE001
            log.exception("gvon-tg: reconnect failed")


async def run_sink(cfg: TgConfig, backfill: int = 0) -> None:
    from telethon import events

    from gvon.telegram_client import make_client, require_authorized

    # one process per session: refuses to start while a pull/login holds it (and blocks them while we run)
    lock = tgc.acquire_session_lock("sink (make telegram)")
    try:
        client = make_client()  # TG_SESSION > GVON_TG_SESSION > data/telegram.session; fails fast on missing
        # TG_API_ID / TG_API_HASH, before the model load
        await require_authorized(client)
        me = await client.get_me()
        log.info("gvon-tg: authorized as Telegram account id=%s username=%s (session: %s)",
                 getattr(me, "id", "?"), getattr(me, "username", None) or "-",
                 "TG_SESSION" if tgc.string_session() else tgc.session_path())
        nul = build_nullifier()
        sink = TelegramSink(client, nul, cfg)
        client.add_event_handler(sink.on_new_message, events.NewMessage(incoming=True))
        log.info("gvon-tg: sink running (dm=%s group=%s group_delete=%s mark_read=%s dry_run=%s); Ctrl-C to stop",
                 cfg.dm_action, cfg.group_action, cfg.group_delete, cfg.mark_read, cfg.dry_run)
        if backfill > 0:
            await sink.backfill(backfill)
        await reconnect_loop(client)
    finally:
        lock.release()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="gvon.telegram_nullifier",
                                 description="Telegram sink: remove nullified incoming messages from your view.")
    ap.add_argument("--vault", nargs="?", type=int, const=20, default=None, metavar="N",
                    help="print the last N vault entries (default 20) and exit; no Telegram connection")
    ap.add_argument("--backfill", type=int, default=0, metavar="N",
                    help="on startup also score the last N unread messages per dialog (default 0 = off)")
    ap.add_argument("--dry-run", action="store_true", help="same as GVON_DRY_RUN=1")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.vault is not None:
        entries = vault_tail(args.vault, VAULT_PATH)
        if not entries:
            print(f"vault {VAULT_PATH} is empty")
        for e in entries:
            print(format_vault_entry(e))
        return 0

    try:
        cfg = TgConfig.from_env(dry_run=True if args.dry_run else None)
    except ValueError as e:
        print(f"gvon-tg: {e}", file=sys.stderr)
        return 2
    try:
        asyncio.run(run_sink(cfg, backfill=args.backfill))
    except tgc.TelegramConfigError as e:
        print(f"gvon-tg: {e}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
