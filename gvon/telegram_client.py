"""Shared Telegram (Telethon) connection helpers: credentials from .env, the session file under data/.

The session file (data/telegram.session) is a FULL login to your Telegram account. It lives under data/,
which is gitignored; this module creates data/ with mode 0700 and tightens the session file to 0600.
Credentials (TG_API_ID / TG_API_HASH, from https://my.telegram.org -> API development tools) are read
via gvon.env and are never printed or logged.

This is the ONLY place a TelegramClient is constructed: gvon.telegram_ingest (login + pull) and
gvon.telegram_nullifier (the sink) both call make_client(). Session resolution, in order:
TG_SESSION (a Telethon StringSession, never written to disk; ignored when allow_string=False, which
`login` uses so it always writes the file) > an explicit session path > GVON_TG_SESSION > data/telegram.session.

Nothing here logs in interactively: the one-time phone + code login is run by the user
(`make telegram-login`, implemented in gvon.telegram_ingest.login). make_client() only builds a client bound
to the existing session; require_authorized() connects and fails with a clear message if that login has
not happened yet. has_creds() / has_session() let `make all` skip the Telegram step without connecting.

Session paths follow Telethon's rule: ".session" is appended when missing (`--session mytg` -> mytg.session),
and every helper here (has_session, secure_session_file, the lock) uses that normalised path.

One process per session: the sink, `pull` and `login` each hold an exclusive flock on <session>.lock
(data/telegram.session.lock, or data/telegram.tg_session.lock for TG_SESSION) for as long as their client
exists. Telethon's SQLite session commits entity rows only about once a minute, so two processes on one
session file fail with "database is locked" (a pull dies in connect(), or the sink drops an update batch).
A second process therefore stops with SessionBusyError, and `ready` reports the holder so `make all`
skips the Telegram pull while the sink runs.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, IO

from gvon import env

DATA_DIR = env.ROOT / "data"
SESSION_SUFFIX = ".session"  # Telethon's SQLiteSession appends this when the name lacks it
DEFAULT_SESSION = DATA_DIR / "telegram.session"
STRING_SESSION_LOCK = DATA_DIR / "telegram.tg_session.lock"
# Telegram's own service account (login codes, new-login and security alerts): never collected by the
# ingest and never acted on by the sink.
SERVICE_USER_IDS = frozenset({777000})


class TelegramConfigError(RuntimeError):
    """Missing/invalid TG_API_ID / TG_API_HASH, or no authorized session. Message never contains secrets."""


class SessionBusyError(TelegramConfigError):
    """Another gvon process (the sink, a pull or a login) holds this session's lock."""


def normalize_session(path: str | Path) -> Path:
    """The file Telethon's SQLiteSession actually uses: ".session" appended when missing."""
    p = Path(path).expanduser()
    return p if p.name.endswith(SESSION_SUFFIX) else p.with_name(p.name + SESSION_SUFFIX)


def session_path() -> Path:
    """GVON_TG_SESSION overrides the default data/telegram.session (must still be kept private)."""
    p = (env.get("GVON_TG_SESSION") or "").strip()
    return normalize_session(p) if p else DEFAULT_SESSION


def load_creds() -> tuple[int, str]:
    """(api_id, api_hash) from the environment / .env. Raises TelegramConfigError naming only the KEY."""
    raw_id = (env.get("TG_API_ID") or "").strip()
    api_hash = (env.get("TG_API_HASH") or "").strip()
    if not raw_id or not api_hash:
        missing = [k for k, v in (("TG_API_ID", raw_id), ("TG_API_HASH", api_hash)) if not v]
        raise TelegramConfigError(
            f"missing {', '.join(missing)} in .env; get them at https://my.telegram.org (API development tools), "
            "see docs/TELEGRAM.md")
    try:
        api_id = int(raw_id)
    except ValueError:
        raise TelegramConfigError("TG_API_ID in .env must be an integer (value not shown)") from None
    return api_id, api_hash


def ensure_private_dir(path: Path) -> None:
    """Create the session's directory with 0700. An existing directory is tightened only when it is the
    repo's data/ directory: never chmod an arbitrary parent such as the current directory or $HOME."""
    path = Path(path)
    existed = path.exists()
    path.mkdir(parents=True, exist_ok=True)
    try:
        if not existed or path.resolve() == DATA_DIR.resolve():
            os.chmod(path, 0o700)
    except OSError:
        pass


def secure_session_file(path: str | Path | None = None) -> None:
    """chmod 0600 the session file (and its sqlite journal, if any). Safe to call when it does not exist."""
    p = normalize_session(path) if path is not None else session_path()
    for f in (p, p.with_name(p.name + "-journal")):
        if f.exists():
            try:
                os.chmod(f, 0o600)
            except OSError:
                pass


def string_session() -> str:
    """The TG_SESSION StringSession from the environment / .env, or "" (value never logged)."""
    return (env.get("TG_SESSION") or "").strip()


def has_creds() -> bool:
    """True when TG_API_ID and TG_API_HASH are both set and the id is an integer (no values exposed)."""
    try:
        load_creds()
    except TelegramConfigError:
        return False
    return True


def has_session(session: str | Path | None = None) -> bool:
    """True when TG_SESSION is set or the session file exists (does NOT check that it is logged in)."""
    if string_session():
        return True
    return (normalize_session(session) if session is not None else session_path()).exists()


def resolve_session(session: str | Path | None = None, *, allow_string: bool = True) -> Any:
    """What TelegramClient gets as its session: a StringSession when TG_SESSION is set (and allowed),
    else the SQLite session file path as str (its directory created 0700)."""
    string = string_session() if allow_string else ""
    if string:
        from telethon.sessions import StringSession

        return StringSession(string)
    sess = normalize_session(session) if session is not None else session_path()
    ensure_private_dir(sess.parent)
    return str(sess)


# ------------------------------------------------------------------------------------------ session lock

def lock_path(session: str | Path | None = None, *, allow_string: bool = True) -> Path:
    """<session file>.lock, or data/telegram.tg_session.lock when TG_SESSION is what the client will use."""
    if allow_string and string_session():
        return STRING_SESSION_LOCK
    sess = normalize_session(session) if session is not None else session_path()
    return sess.with_name(sess.name + ".lock")


class SessionLock:
    """An exclusive, non-blocking fcntl.flock held for the life of one Telegram client. The lock file holds
    only "<holder> pid=<pid>" (no secrets). The OS drops the lock when the process exits, even on a crash."""

    def __init__(self, path: Path, holder: str) -> None:
        self.path = Path(path)
        self.holder = holder
        self._fh: IO[str] | None = None

    def acquire(self) -> "SessionLock":
        import fcntl

        ensure_private_dir(self.path.parent)
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o600)
        fh = os.fdopen(fd, "r+", encoding="utf-8")
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            other = _read_holder(fh)
            fh.close()
            raise SessionBusyError(
                f"Telegram session is in use by another gvon process ({other or 'unknown holder'}; lock "
                f"{self.path}). Telethon's SQLite session cannot be shared: stop that process first "
                "(the sink is `make telegram`)") from None
        fh.seek(0)
        fh.truncate()
        fh.write(f"{self.holder} pid={os.getpid()}\n")
        fh.flush()
        self._fh = fh
        return self

    def release(self) -> None:
        if self._fh is not None:
            import fcntl

            try:
                fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
            finally:
                self._fh.close()
                self._fh = None

    def __enter__(self) -> "SessionLock":
        return self if self._fh is not None else self.acquire()  # acquire_session_lock() already holds it

    def __exit__(self, *exc: Any) -> None:
        self.release()


def _read_holder(fh: IO[str]) -> str:
    try:
        fh.seek(0)
        return fh.read(200).strip()
    except OSError:
        return ""


def acquire_session_lock(holder: str, session: str | Path | None = None, *,
                         allow_string: bool = True) -> SessionLock:
    """Take the session's exclusive lock or raise SessionBusyError (a TelegramConfigError) naming the holder."""
    return SessionLock(lock_path(session, allow_string=allow_string), holder).acquire()


def session_in_use(session: str | Path | None = None, *, allow_string: bool = True) -> str | None:
    """The holder string ("sink pid=123") when another process holds the session lock, else None.
    Never creates the lock file and never blocks."""
    import fcntl

    path = lock_path(session, allow_string=allow_string)
    if not path.exists():
        return None
    try:
        fh = open(path, "r", encoding="utf-8")
    except OSError:
        return None
    try:
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
        except OSError:
            return _read_holder(fh) or "another gvon process"
        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        return None
    finally:
        fh.close()


def make_client(session: str | Path | None = None, *, allow_string: bool = True, api_id: int | None = None,
                api_hash: str | None = None, **kwargs: Any) -> Any:
    """The one telethon.TelegramClient factory (not connected, not logged in).

    session: SQLite session file (default GVON_TG_SESSION, else data/telegram.session).
    allow_string: let TG_SESSION (StringSession) take precedence; `login` passes False.
    Extra kwargs go to TelegramClient (e.g. flood_sleep_threshold, receive_updates)."""
    from telethon import TelegramClient  # lazy: keep `import gvon.telegram_client` cheap

    if api_id is None or api_hash is None:
        api_id, api_hash = load_creds()
    resolved = resolve_session(session, allow_string=allow_string)
    kwargs.setdefault("flood_sleep_threshold", 60)  # Telethon auto-sleeps FloodWaits shorter than this
    return TelegramClient(resolved, api_id, api_hash, **kwargs)


async def require_authorized(client: Any, session: str | Path | None = None) -> None:
    """Connect and ensure the session is logged in; never prompts for phone/code."""
    if not client.is_connected():
        await client.connect()
    if not await client.is_user_authorized():
        await client.disconnect()
        where = "TG_SESSION" if string_session() else str(normalize_session(session) if session is not None
                                                           else session_path())
        raise TelegramConfigError(
            f"Telegram session {where} is not logged in; run `make telegram-login` once (interactive)")
    secure_session_file(session)


def readiness(session: str | Path | None = None) -> tuple[bool, str]:
    """(ready, reason) for running a Telegram step unattended: creds present, a session to use, AND no other
    gvon process (e.g. the always-on sink) holding that session's lock.
    Never connects to Telegram and never includes credential or session values in the reason."""
    try:
        load_creds()
    except TelegramConfigError as e:
        return False, str(e)
    p = normalize_session(session) if session is not None else session_path()
    if not has_session(session):
        return False, f"no Telegram session at {p} (and TG_SESSION unset); run `make telegram-login` once (interactive)"
    holder = session_in_use(session)
    if holder:
        return False, (f"the Telegram session is in use by {holder} (the sink `make telegram`?); one session cannot "
                       "serve two processes, so this run skips the Telegram pull. Stop the sink and run "
                       "`make telegram-ingest`, then restart the sink")
    return True, "TG_SESSION" if string_session() else str(p)


def main(argv: list[str] | None = None) -> int:
    """`python -m gvon.telegram_client ready`: exit 0 if creds + a free session are present, 3 otherwise
    (missing creds/session, or the session is held by another gvon process such as the sink). Used by
    `make all` / scripts/run-all.sh to skip the Telegram step with a clear message."""
    import argparse
    import sys

    ap = argparse.ArgumentParser(prog="python -m gvon.telegram_client")
    ap.add_argument("cmd", choices=["ready"])
    ap.add_argument("--session", default=None)
    args = ap.parse_args(argv)
    ok, reason = readiness(args.session)
    if ok:
        print(f"telegram: ready (session: {reason})")
        return 0
    print(f"telegram: not configured: {reason}", file=sys.stderr)
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
