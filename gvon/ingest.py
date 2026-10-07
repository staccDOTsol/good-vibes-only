"""Ingest every X engagement with GVON_HANDLE from the last N days via X API v2 recent search.

Why this shape:
- Recent search (app-only bearer) is the only v2 read path that covers replies, mentions and quotes
  from arbitrary accounts, so we run four queries against it and merge the results.
- Raw API pages are appended to data/raw/_pages/<query>.jsonl as they arrive and per-query cursors
  live in data/raw/_state.json. A 429 or crash therefore costs nothing: re-running continues from the
  saved next_token, and the final JSONL files are always rebuilt from every page on disk.
- Recent search only accepts a start_time inside the last 7 days, and a next_token is bound to the
  request's start_time. A run is started at now-7d+2min, so after a few minutes (a 429 sleep, a 402
  top-up, a crash) that start_time is invalid. Instead of failing, the window "rolls": start_time is
  recomputed and every unfinished query continues WITHOUT its cursor but with end_time = the oldest
  tweet it already has (+1s), so already-fetched pages are kept (merge dedups by id) and not re-billed.
  A 400 that names start_time triggers the same roll; it is never mistaken for an operator rejection.
- A completed run younger than --max-age-hours (default 6) is reused instead of re-pulled, so re-running
  `make all` after a later stage failed does not delete the pages and re-bill the whole week.
  --refresh forces a new window; --fresh also discards unfinished state.
- Normalization is pure (no I/O) so it can be unit-tested on synthetic payloads.

Outputs (see CONTRACT.md): data/raw/tweets.jsonl, users.jsonl, context.jsonl.
The bearer token is never printed, logged or written; error bodies are redacted before display.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import requests

from gvon import env

API = "https://api.x.com/2"
SEARCH_URL = f"{API}/tweets/search/recent"
COUNTS_URL = f"{API}/tweets/counts/recent"

TWEET_FIELDS = (
    "id,text,author_id,created_at,conversation_id,in_reply_to_user_id,lang,"
    "public_metrics,referenced_tweets,note_tweet"
)
EXPANSIONS = "author_id,referenced_tweets.id,referenced_tweets.id.author_id,in_reply_to_user_id"
USER_FIELDS = "id,username,name,description,created_at,public_metrics,verified"

MAX_RATE_WAIT_S = 16 * 60  # cap any single 429 sleep at 16 min
MAX_RATE_WAITS = 6  # after this many 429 sleeps in one run, persist state and exit for a later resume
RETRIES_5XX = 3
# Recent search rejects start_time older than 7 days; keep a margin for clock skew + request latency.
SEARCH_WINDOW_MARGIN = timedelta(minutes=2)
STALE_SAFETY = timedelta(seconds=30)  # roll the window once start_time is this close to the 7-day floor
MAX_WINDOW_ROLLS_PER_QUERY = 3  # repeated start_time 400s for one query => give up (clock badly off?)
DEFAULT_MAX_AGE_HOURS = 6.0
DEFAULT_OUT = env.ROOT / "data" / "raw"  # repo-anchored: real tweets never land outside gitignored data/

# Higher wins when the same tweet id shows up from several queries.
KIND_PRECEDENCE = {"own": 4, "reply": 3, "quote": 2, "mention": 1}

log = logging.getLogger("gvon.ingest")


@dataclass(frozen=True)
class QuerySpec:
    name: str
    query: str
    own: bool  # True => every tweet from this query is kind="own"
    optional: bool  # True => drop (and note) if the API rejects the operator


def build_queries(handle: str) -> list[QuerySpec]:
    """The four pooled queries from the spec: mentions, replies, quotes (url:), own tweets."""
    h = handle.lstrip("@")
    return [
        QuerySpec("A_mentions", f"@{h} -from:{h} -is:retweet", own=False, optional=False),
        QuerySpec("B_replies", f"to:{h} -from:{h} -is:retweet", own=False, optional=False),
        QuerySpec("C_quotes", f"url:x.com/{h} -from:{h} -is:retweet", own=False, optional=True),
        QuerySpec("D_own", f"from:{h}", own=True, optional=False),
    ]


# --------------------------------------------------------------------------- normalization (pure)


def classify_kind(tweet: dict[str, Any], *, own: bool) -> str:
    """own for query D; else reply if it replies to anything, quote if it quotes, mention otherwise."""
    if own:
        return "own"
    types = {r.get("type") for r in tweet.get("referenced_tweets") or []}
    if "replied_to" in types:
        return "reply"
    if "quoted" in types:
        return "quote"
    return "mention"


def full_text(tweet: dict[str, Any]) -> str:
    """Long posts arrive truncated in `text`; note_tweet.text carries the whole thing."""
    note = tweet.get("note_tweet") or {}
    return note.get("text") or tweet.get("text") or ""


def normalize_tweet(tweet: dict[str, Any], kind: str, users: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Map an API v2 tweet object to the CONTRACT.md tweets.jsonl row."""
    author = users.get(str(tweet.get("author_id")), {})
    return {
        "id": str(tweet["id"]),
        "text": full_text(tweet),
        "author_id": str(tweet.get("author_id")) if tweet.get("author_id") is not None else None,
        "author_username": author.get("username"),
        "author_name": author.get("name"),
        "created_at": tweet.get("created_at"),
        "conversation_id": tweet.get("conversation_id"),
        "in_reply_to_user_id": tweet.get("in_reply_to_user_id"),
        "lang": tweet.get("lang"),
        "kind": kind,
        "referenced": [
            {"type": r.get("type"), "id": str(r.get("id"))} for r in tweet.get("referenced_tweets") or []
        ],
        "public_metrics": dict(tweet.get("public_metrics") or {}),
    }


def normalize_user(user: dict[str, Any]) -> dict[str, Any]:
    """Map an API v2 user object to the CONTRACT.md users.jsonl row."""
    pm = user.get("public_metrics") or {}
    return {
        "id": str(user["id"]),
        "username": user.get("username"),
        "name": user.get("name"),
        "description": user.get("description"),
        "created_at": user.get("created_at"),
        "public_metrics": {
            "followers_count": pm.get("followers_count"),
            "following_count": pm.get("following_count"),
            "tweet_count": pm.get("tweet_count"),
        },
        "verified": bool(user.get("verified", False)),
    }


def merge_pages(pages: Iterable[tuple[bool, dict[str, Any]]]) -> tuple[list[dict], list[dict], list[dict]]:
    """Merge raw search pages into (tweets, users, context), deduped by id.

    `pages` yields (own, response_json). Users are collected first across all pages so author
    lookups work even when the author object arrived on a different page/query.
    """
    pages = list(pages)
    raw_users: dict[str, dict[str, Any]] = {}
    for _, page in pages:
        for u in (page.get("includes") or {}).get("users") or []:
            raw_users.setdefault(str(u["id"]), u)

    tweets: dict[str, dict[str, Any]] = {}
    included: dict[str, dict[str, Any]] = {}
    for own, page in pages:
        for t in page.get("data") or []:
            kind = classify_kind(t, own=own)
            tid = str(t["id"])
            prev = tweets.get(tid)
            if prev is None or KIND_PRECEDENCE[kind] > KIND_PRECEDENCE[prev["kind"]]:
                tweets[tid] = normalize_tweet(t, kind, raw_users)
        for t in (page.get("includes") or {}).get("tweets") or []:
            included.setdefault(str(t["id"]), t)

    context = [
        normalize_tweet(t, "context", raw_users) for tid, t in included.items() if tid not in tweets
    ]
    users = [normalize_user(u) for u in raw_users.values()]
    by_time = lambda r: (r.get("created_at") or "", r["id"])  # noqa: E731
    return sorted(tweets.values(), key=by_time), users, sorted(context, key=by_time)


# --------------------------------------------------------------------------- HTTP


class ApiError(RuntimeError):
    def __init__(self, status: int, body: str):
        super().__init__(f"HTTP {status}: {body}")
        self.status = status
        self.body = body


class RateLimitExhausted(RuntimeError):
    pass


class Client:
    """Bearer-auth session with 429 sleep (capped) and 5xx retry. Never exposes the token."""

    def __init__(self, token: str, max_rate_waits: int = MAX_RATE_WAITS):
        self._token = token
        self._s = requests.Session()
        self._s.headers["Authorization"] = f"Bearer {token}"
        self._s.headers["User-Agent"] = "gvon-ingest/0.1"
        self.rate_waits = 0
        self.max_rate_waits = max_rate_waits

    def redact(self, s: str) -> str:
        return s.replace(self._token, "<REDACTED>") if self._token else s

    def get(self, url: str, params: dict[str, Any] | Callable[[], dict[str, Any]]) -> dict[str, Any]:
        """GET with retries. `params` may be a callable so it is rebuilt after every wait: a 429 sleep can
        push a search start_time out of the 7-day window, and the caller's builder rolls it forward."""
        attempt_5xx = 0
        while True:
            try:
                r = self._s.get(url, params=params() if callable(params) else params, timeout=30)
            except requests.RequestException as e:
                attempt_5xx += 1
                if attempt_5xx > RETRIES_5XX:
                    raise ApiError(-1, self.redact(f"network error: {type(e).__name__}: {e}")) from None
                self._backoff(attempt_5xx, f"network error {type(e).__name__}")
                continue
            if r.status_code == 200:
                return r.json()
            if r.status_code == 429:
                self._wait_rate_limit(r)
                continue
            if 500 <= r.status_code < 600:
                attempt_5xx += 1
                if attempt_5xx > RETRIES_5XX:
                    raise ApiError(r.status_code, self.redact(r.text))
                self._backoff(attempt_5xx, f"HTTP {r.status_code}")
                continue
            raise ApiError(r.status_code, self.redact(r.text))

    def _backoff(self, attempt: int, why: str) -> None:
        delay = 2 ** attempt
        log.warning("%s; retry %d/%d in %ds", why, attempt, RETRIES_5XX, delay)
        time.sleep(delay)

    def _wait_rate_limit(self, r: requests.Response) -> None:
        if self.rate_waits >= self.max_rate_waits:
            raise RateLimitExhausted(f"hit {self.rate_waits} rate-limit waits; state saved, re-run to resume")
        self.rate_waits += 1
        reset = r.headers.get("x-rate-limit-reset")
        wait = 60.0
        if reset and reset.isdigit():
            wait = max(1.0, int(reset) - time.time() + 1)
        wait = min(wait, MAX_RATE_WAIT_S)
        log.warning(
            "429 rate limited (remaining=%s, reset=%s); sleeping %.0fs (wait %d/%d)",
            r.headers.get("x-rate-limit-remaining"), reset, wait, self.rate_waits, self.max_rate_waits,
        )
        time.sleep(wait)


# --------------------------------------------------------------------------- state + pages on disk


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def _parse_ts(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        return None


def is_stale(start_time: str, now: datetime | None = None) -> bool:
    """True once start_time is (nearly) older than recent search's 7-day floor."""
    now = now or datetime.now(timezone.utc)
    st = _parse_ts(start_time)
    return st is None or st < now - timedelta(days=7) + STALE_SAFETY


def oldest_in_page(page: dict[str, Any]) -> str | None:
    """Oldest created_at among a page's primary results (pages arrive newest-first)."""
    times = [t.get("created_at") for t in page.get("data") or [] if t.get("created_at")]
    return min(times, key=lambda x: _parse_ts(x) or datetime.max.replace(tzinfo=timezone.utc)) if times else None


def roll_window(state: dict, now: datetime | None = None, reason: str = "start_time left the 7-day window") -> None:
    """Move start_time forward and continue every unfinished query from its oldest tweet without a cursor.

    end_time = oldest fetched created_at + 1s (re-fetches that second's tweets at most; merge dedups), so
    already-fetched pages are kept and not paid for again. A query whose end_time is already at or before
    the new start_time has nothing left in range and is marked completed.
    """
    now = now or datetime.now(timezone.utc)
    new_start = compute_start_time(float(state.get("since_days") or 7.0), now)
    log.warning("%s (start_time=%s); continuing with start_time=%s and per-query end_time", reason,
                state.get("start_time"), new_start)
    state["start_time"] = new_start
    state["window_rolls"] = int(state.get("window_rolls") or 0) + 1
    start_dt = _parse_ts(new_start)
    for qs in state["queries"].values():
        if qs.get("completed"):
            continue
        qs["next_token"] = None
        oldest = _parse_ts(qs.get("oldest_created_at"))
        if oldest is None:
            qs["end_time"] = None  # nothing fetched yet: plain restart, nothing to re-bill
            continue
        end = oldest + timedelta(seconds=1)
        if start_dt is not None and end <= start_dt:
            qs.update(completed=True, end_time=None)
        else:
            qs["end_time"] = _iso(end)


def _backfill_oldest(out: Path, state: dict) -> None:
    """States written before oldest_created_at existed: derive it from the saved pages."""
    for name, qs in state["queries"].items():
        if qs.get("oldest_created_at") or qs.get("completed"):
            continue
        f = out / "_pages" / f"{name}.jsonl"
        if not f.exists():
            continue
        olds = [o for o in (oldest_in_page(json.loads(line)) for line in f.read_text().splitlines() if line.strip()) if o]
        if olds:
            qs["oldest_created_at"] = min(olds, key=lambda x: _parse_ts(x) or datetime.max.replace(tzinfo=timezone.utc))


def recent_completed_state(out: Path, handle: str, specs: list[QuerySpec], max_age_hours: float,
                           now: datetime | None = None) -> dict | None:
    """The saved state if it is a completed run for this handle finished less than max_age_hours ago."""
    path = out / "_state.json"
    if not path.exists():
        return None
    try:
        state = json.loads(path.read_text())
    except json.JSONDecodeError:
        return None
    qs = state.get("queries", {})
    if state.get("handle") != handle or not {s.name for s in specs} <= set(qs) or any(not q.get("completed") for q in qs.values()):
        return None
    now = now or datetime.now(timezone.utc)
    finished = _parse_ts(state.get("finished_at")) or datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)
    return state if now - finished < timedelta(hours=max_age_hours) else None


def load_state(out: Path, specs: list[QuerySpec], start_time: str, handle: str, fresh: bool,
               since_days: float = 7.0, now: datetime | None = None) -> dict:
    """Reuse an unfinished run's cursors; roll its window first if its start_time has gone stale."""
    path = out / "_state.json"
    if path.exists() and not fresh:
        state = json.loads(path.read_text())
        qs = state.get("queries", {})
        unfinished = any(not q.get("completed") for q in qs.values())
        # A fully completed state is a finished run: start a new window instead of reusing it.
        if state.get("handle") == handle and {s.name for s in specs} <= set(qs) and unfinished:
            state.setdefault("since_days", since_days)
            _backfill_oldest(out, state)
            log.info("resuming from %s (start_time=%s)", path, state["start_time"])
            if is_stale(state["start_time"], now):
                roll_window(state, now, "saved start_time is outside recent search's 7-day window")
            return state
    pages_dir = out / "_pages"
    if pages_dir.exists():
        for f in pages_dir.glob("*.jsonl"):
            f.unlink()
    return {
        "handle": handle,
        "start_time": start_time,
        "since_days": since_days,
        "created_at": _iso(datetime.now(timezone.utc)),
        "queries": {
            s.name: {"query": s.query, "next_token": None, "completed": False, "dropped": None, "pages": 0, "tweets": 0,
                     "end_time": None, "oldest_created_at": None}
            for s in specs
        },
    }


def save_state(out: Path, state: dict) -> None:
    _atomic_write(out / "_state.json", json.dumps(state, indent=2))


def append_page(out: Path, name: str, page: dict) -> None:
    d = out / "_pages"
    d.mkdir(parents=True, exist_ok=True)
    with (d / f"{name}.jsonl").open("a") as f:
        f.write(json.dumps(page) + "\n")


def read_pages(out: Path, specs: list[QuerySpec]) -> list[tuple[bool, dict]]:
    pages: list[tuple[bool, dict]] = []
    for s in specs:
        f = out / "_pages" / f"{s.name}.jsonl"
        if f.exists():
            pages += [(s.own, json.loads(line)) for line in f.read_text().splitlines() if line.strip()]
    return pages


def write_jsonl(path: Path, rows: list[dict]) -> None:
    _atomic_write(path, "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


# --------------------------------------------------------------------------- fetch loop


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def compute_start_time(since_days: float, now: datetime | None = None) -> str:
    """now - since_days, clamped inside recent search's 7-day window."""
    now = now or datetime.now(timezone.utc)
    floor = now - timedelta(days=7) + SEARCH_WINDOW_MARGIN
    return _iso(max(now - timedelta(days=since_days), floor))


def search_params(q: QuerySpec, start_time: str, next_token: str | None, end_time: str | None = None) -> dict[str, Any]:
    p: dict[str, Any] = {
        "query": q.query,
        "start_time": start_time,
        "max_results": 100,
        "tweet.fields": TWEET_FIELDS,
        "expansions": EXPANSIONS,
        "user.fields": USER_FIELDS,
    }
    if end_time:
        p["end_time"] = end_time
    if next_token:
        p["next_token"] = next_token
    return p


def _names_start_time(e: ApiError) -> bool:
    return e.status == 400 and "start_time" in (e.body or "")


def fetch_all(client: Client, specs: list[QuerySpec], state: dict, out: Path) -> None:
    """Round-robin one page per active query until every query is exhausted (or dropped)."""
    active = [s for s in specs if not state["queries"][s.name]["completed"]]
    start_time_400s: dict[str, int] = {}
    while active:
        for s in list(active):
            qs = state["queries"][s.name]
            if qs["completed"]:  # a window roll can finish a query whose range is used up
                active.remove(s)
                continue

            def params(s: QuerySpec = s, qs: dict = qs) -> dict[str, Any]:
                if is_stale(state["start_time"]):
                    roll_window(state)
                    save_state(out, state)
                return search_params(s, state["start_time"], qs["next_token"], qs.get("end_time"))

            try:
                page = client.get(SEARCH_URL, params)
            except ApiError as e:
                if _names_start_time(e):
                    # never an operator rejection: the window moved under us (clock skew / long wait)
                    start_time_400s[s.name] = start_time_400s.get(s.name, 0) + 1
                    if start_time_400s[s.name] > MAX_WINDOW_ROLLS_PER_QUERY:
                        raise
                    roll_window(state, datetime.now(timezone.utc) + timedelta(minutes=5),
                                f"HTTP 400 about start_time for {s.name}")
                    save_state(out, state)
                    continue
                if s.optional and e.status == 400 and qs["pages"] == 0:
                    log.warning("query %s rejected (HTTP 400), dropping: %s", s.name, e.body[:500])
                    qs.update(completed=True, dropped=f"HTTP 400: {e.body[:1000]}")
                    save_state(out, state)
                    active.remove(s)
                    continue
                raise
            append_page(out, s.name, page)
            meta = page.get("meta") or {}
            oldest = oldest_in_page(page)
            if oldest and (not qs.get("oldest_created_at")
                           or (_parse_ts(oldest) or datetime.max.replace(tzinfo=timezone.utc))
                           < (_parse_ts(qs["oldest_created_at"]) or datetime.max.replace(tzinfo=timezone.utc))):
                qs["oldest_created_at"] = oldest
            qs["pages"] += 1
            qs["tweets"] += int(meta.get("result_count") or 0)
            qs["next_token"] = meta.get("next_token")
            if not qs["next_token"]:
                qs["completed"] = True
                active.remove(s)
            save_state(out, state)
            log.info(
                "%s page %d: +%s (total %d)%s", s.name, qs["pages"], meta.get("result_count"),
                qs["tweets"], "" if qs["next_token"] else " [done]",
            )


def dry_run(client: Client, specs: list[QuerySpec], start_time: str) -> dict[str, Any]:
    """Counts-only: hit /tweets/counts/recent so nothing is downloaded or written."""
    counts: dict[str, Any] = {}
    for s in specs:
        try:
            total, token = 0, None
            while True:
                p: dict[str, Any] = {"query": s.query, "start_time": start_time, "granularity": "day"}
                if token:
                    p["next_token"] = token
                r = client.get(COUNTS_URL, p)
                total += int((r.get("meta") or {}).get("total_tweet_count") or 0)
                token = (r.get("meta") or {}).get("next_token")
                if not token:
                    break
            counts[s.name] = total
        except ApiError as e:
            counts[s.name] = f"error HTTP {e.status}: {e.body[:500]}"
    return counts


def summarize(tweets: list[dict], users: list[dict], context: list[dict], state: dict) -> dict[str, Any]:
    kinds: dict[str, int] = {}
    for t in tweets:
        kinds[t["kind"]] = kinds.get(t["kind"], 0) + 1
    engagers = {t["author_id"] for t in tweets if t["kind"] != "own"}
    dates = sorted(t["created_at"] for t in tweets if t.get("created_at"))
    return {
        "tweets": len(tweets),
        "by_kind": kinds,
        "unique_engaging_authors": len(engagers),
        "users": len(users),
        "context": len(context),
        "date_range": [dates[0], dates[-1]] if dates else None,
        "start_time": state["start_time"],
        "queries": {
            k: {kk: v[kk] for kk in ("query", "pages", "tweets", "completed", "dropped")}
            for k, v in state["queries"].items()
        },
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--since-days", type=float, default=7.0, help="lookback window (clamped to 7d)")
    ap.add_argument("--dry-run", action="store_true", help="print per-query counts only; write nothing")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="output directory (default: <repo>/data/raw)")
    ap.add_argument("--fresh", action="store_true", help="ignore _state.json, delete saved pages and start over")
    ap.add_argument("--refresh", action="store_true",
                    help="re-pull a new 7-day window even if a completed run is younger than --max-age-hours")
    ap.add_argument("--max-age-hours", type=float, default=DEFAULT_MAX_AGE_HOURS,
                    help="reuse a completed run younger than this instead of re-pulling (default 6)")
    ap.add_argument("--handle", default=None, help="override GVON_HANDLE")
    ap.add_argument("--rebuild-only", action="store_true", help="rebuild JSONL from saved pages; no API calls")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    # urllib3 debug logs would include request lines; keep them quiet regardless of -v.
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    token = env.get("X_BEARER_TOKEN")
    handle = (args.handle or env.get("GVON_HANDLE") or "").lstrip("@")
    if not token:
        print("X_BEARER_TOKEN missing from .env", file=sys.stderr)
        return 2
    if not handle:
        print("GVON_HANDLE missing from .env (or pass --handle)", file=sys.stderr)
        return 2

    specs = build_queries(handle)
    client = Client(token)
    start_time = compute_start_time(args.since_days)

    if args.dry_run:
        print(json.dumps({"handle": handle, "start_time": start_time, "counts": dry_run(client, specs, start_time)}, indent=2))
        return 0

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.rebuild_only:
        state = json.loads((out / "_state.json").read_text())
        return build_outputs(out, specs, state)
    if not (args.fresh or args.refresh):
        recent = recent_completed_state(out, handle, specs, args.max_age_hours)
        if recent is not None:
            print(f"ingest: a completed run (start_time={recent['start_time']}) is younger than {args.max_age_hours}h; "
                  "reusing it without API calls (pass --refresh to re-pull)", file=sys.stderr)
            return build_outputs(out, specs, recent)
    state = load_state(out, specs, start_time, handle, args.fresh, args.since_days)
    save_state(out, state)
    rc = 0
    try:
        fetch_all(client, specs, state, out)
    except RateLimitExhausted as e:
        print(f"PAUSED: {e}", file=sys.stderr)
        rc = 3
    except ApiError as e:
        print(f"FAILED: HTTP {e.status}\n{e.body}", file=sys.stderr)
        state["last_error"] = {"status": e.status, "body": e.body[:2000]}
        save_state(out, state)
        rc = 1
    if all(q["completed"] for q in state["queries"].values()) and not state.get("finished_at"):
        state["finished_at"] = _iso(datetime.now(timezone.utc))
        save_state(out, state)
    # Always materialize whatever real pages are on disk; summary.partial flags an incomplete run.
    build_outputs(out, specs, state)
    return rc


def build_outputs(out: Path, specs: list[QuerySpec], state: dict) -> int:
    """Rebuild tweets/users/context JSONL from every saved page (no network)."""
    tweets, users, context = merge_pages(read_pages(out, specs))
    write_jsonl(out / "tweets.jsonl", tweets)
    write_jsonl(out / "users.jsonl", users)
    write_jsonl(out / "context.jsonl", context)
    summary = summarize(tweets, users, context, state)
    summary["partial"] = any(not q["completed"] for q in state["queries"].values())
    _atomic_write(out / "_summary.json", json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
