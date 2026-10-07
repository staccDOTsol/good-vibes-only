"""Pure JSON pruning for x.com timeline responses: remove nullified posts before the browser sees them.

Why this module exists separately from the mitmproxy addon (gvon/nullifier.py): the pruning logic is
the part most likely to break when X ships a new web build, so it has no mitmproxy import and is unit
tested against synthetic fixtures (fixtures/*.json) and, later, real recordings (data/recordings/).

Design goal: be structure-agnostic. X renames GraphQL operations (HomeTimeline, HomeLatestTimeline,
TweetDetail, SearchTimeline, NotificationsTimeline, ...) and moves the `instructions` array around
inside `data`, so nothing here keys off operation names or exact paths. Instead:

* A *tweet result* is any dict with `legacy.full_text` or `note_tweet` (GraphQL Tweet objects; the
  `TweetWithVisibilityResults` wrapper is transparent because its inner `.tweet` matches too).
* A *user-only item* is any `user_results` dict that is NOT inside a tweet result (who-to-follow cells,
  GraphQL user notifications, aggregate `from_users`). It is judged by identity only (blocklist), never
  by text. Users nested inside someone else's tweet (the author under `core`, a media attribution
  `additional_media_info.source_user`, ...) are NOT user-only items: a friend's post is not dropped
  because its clip originally came from a blocklisted account.
* An *entry* is any list element dict with an `entryId` (timeline entries and module items alike, and
  `TimelineAddToModule.moduleItems`). An entry whose subtree still contains a nullified tweet/user after
  its children were pruned is dropped. A module entry whose *content* items all got dropped is dropped
  too, even when a module-level cursor item ("Show replies", entryId `...-cursor-showmore-...`) remains,
  because that cursor would only expand the nullified thread.
  Instructions holding a single `entry` (TimelinePinEntry, TimelineReplaceEntry) are dropped as a whole.
* Cursor entries (`entryId` starting with "cursor-") are never touched, so infinite scroll keeps working.
* Text scope: a tweet's TEXT is sent to the decider only when the tweet engages the owner (replies to,
  mentions or quotes the owner) unless text_scope="all". The student model was trained only on
  engagement aimed at the owner, so scoring arbitrary home-timeline/search posts is out of distribution
  and would silently hide posts from accounts the owner follows. Every tweet is still checked against
  the blocklist (identity), whatever the scope.
* An entry whose primary tweet is authored by the account owner is never dropped (the focal tweet of
  your own TweetDetail page stays even if it quotes something nullified), and owner-authored tweets are
  never sent to the decider at all.

The legacy REST `globalObjects` shape (/i/api/2/notifications/all.json) is handled by a dedicated pass,
because there entries reference tweets/users/notifications by id instead of embedding them. That pass
propagates "quotes/retweets a nullified tweet" to a fixed point, drops any aggregate notification that
names a blocklisted user (the same policy as the GraphQL path, which drops the whole entry) and never
deletes a tweet or user object that a kept tweet or kept notification still references.

Totality: prune() never raises. On any exception it logs and returns the original payload object
unchanged with an empty report, so a parsing bug degrades to "nothing filtered", never to a broken page.

The decider is any object with the gvon.classifier.Nullifier interface:
`should_nullify(text, *, author_id=None, username=None) -> bool` and optionally `score(texts)` (used only
to batch-warm the classifier cache) and `is_blocked(*, author_id=None, username=None)` (used for
user-only items; when absent we call `should_nullify("", ...)`).

Author context: when the decider's should_nullify declares an `author` parameter (and score() an
`authors` parameter), each judged tweet also carries its author's cheap context, read from the same
payload: {bio, followers, following, tweet_count, created_at, verified} (graphql_author / rest_author).
Fields the payload lacks are None (unknown). Deciders without those parameters are called exactly as before.
"""
from __future__ import annotations

import argparse
import inspect
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, MutableMapping, Protocol

log = logging.getLogger("gvon.prune")

CURSOR_PREFIX = "cursor-"

# A report row: {"entryId": str, "username": str | None, "reason": str}
ReportRow = dict[str, Any]


class Decider(Protocol):
    """The subset of gvon.classifier.Nullifier that prune() needs."""

    def should_nullify(self, text: str, *, author_id: str | None = None, username: str | None = None) -> bool: ...


@dataclass(frozen=True)
class Hit:
    """One nullified thing found inside a subtree (bubbles up until an entry consumes it)."""

    kind: str  # "tweet" | "user"
    id: str | None
    username: str | None
    reason: str


# ---------------------------------------------------------------------------------------------------
# Shape helpers (all tolerant of missing keys / wrong types)
# ---------------------------------------------------------------------------------------------------


def _dict(x: Any) -> dict[str, Any]:
    return x if isinstance(x, dict) else {}


def _path(d: Any, *keys: str) -> Any:
    """Follow dict keys, returning None as soon as something is missing or not a dict."""
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur


def looks_like_tweet(d: dict[str, Any]) -> bool:
    """GraphQL Tweet result: has legacy.full_text (normal posts) or note_tweet (long posts)."""
    legacy = d.get("legacy")
    if isinstance(legacy, dict) and "full_text" in legacy:
        return True
    return isinstance(d.get("note_tweet"), dict)


def tweet_text(t: dict[str, Any]) -> str:
    """Full text of a tweet result; long posts carry the untruncated text in note_tweet."""
    note = _path(t, "note_tweet", "note_tweet_results", "result", "text")
    if isinstance(note, str) and note:
        return note
    full = _path(t, "legacy", "full_text")
    return full if isinstance(full, str) else ""


def tweet_id(t: dict[str, Any]) -> str | None:
    tid = t.get("rest_id") or _path(t, "legacy", "id_str")
    return str(tid) if tid not in (None, "") else None


def user_identity(u: Any) -> tuple[str | None, str | None]:
    """(rest_id, screen_name) of a GraphQL user result; newer builds move screen_name to .core."""
    u = _dict(u)
    uid = u.get("rest_id")
    name = _path(u, "core", "screen_name") or _path(u, "legacy", "screen_name")
    return (str(uid) if uid not in (None, "") else None, name if isinstance(name, str) else None)


def _first(*values: Any) -> Any:
    """First value that is not None (0 and "" count as present)."""
    return next((v for v in values if v is not None), None)


def _verified(*flags: Any) -> bool | None:
    """True if any verification flag is true, False if some flag is present and false, None if none is present."""
    present = [f for f in flags if isinstance(f, bool)]
    return any(present) if present else None


def graphql_author(u: Any) -> dict[str, Any] | None:
    """Author context of a GraphQL user result: legacy.description (or profile_bio.description),
    legacy.followers_count / friends_count / statuses_count, legacy.created_at (newer builds: core.created_at),
    is_blue_verified / legacy.verified (/ verification.verified). None when there is no user object."""
    u = _dict(u)
    if not u:
        return None
    legacy = _dict(u.get("legacy"))
    return {
        "bio": _first(legacy.get("description"), _path(u, "profile_bio", "description")),
        "followers": legacy.get("followers_count"),
        "following": legacy.get("friends_count"),
        "tweet_count": legacy.get("statuses_count"),
        "created_at": _first(legacy.get("created_at"), _path(u, "core", "created_at")),
        "verified": _verified(u.get("is_blue_verified"), legacy.get("verified"), _path(u, "verification", "verified")),
    }


def rest_author(u: Any) -> dict[str, Any] | None:
    """Author context of a legacy REST user object (globalObjects.users[id])."""
    u = _dict(u)
    if not u:
        return None
    return {
        "bio": u.get("description"),
        "followers": u.get("followers_count"),
        "following": u.get("friends_count"),
        "tweet_count": u.get("statuses_count"),
        "created_at": u.get("created_at"),
        "verified": _verified(u.get("is_blue_verified"), u.get("verified"), u.get("ext_is_blue_verified")),
    }


def tweet_author_context(t: dict[str, Any]) -> dict[str, Any] | None:
    """graphql_author of the tweet's author (core.user_results.result), or None."""
    return graphql_author(_path(t, "core", "user_results", "result"))


def _accepts(fn: Any, name: str) -> bool:
    """True when callable `fn` declares a parameter called `name` (explicitly, not just **kwargs)."""
    try:
        return name in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


def tweet_author(t: dict[str, Any]) -> tuple[str | None, str | None]:
    uid, name = user_identity(_path(t, "core", "user_results", "result"))
    if uid is None:
        fallback = _path(t, "legacy", "user_id_str")
        uid = str(fallback) if fallback not in (None, "") else None
    return uid, name


def _unwrap_visibility(r: Any) -> Any:
    """TweetWithVisibilityResults wraps the real Tweet in .tweet."""
    if isinstance(r, dict) and isinstance(r.get("tweet"), dict) and not looks_like_tweet(r):
        return r["tweet"]
    return r


def primary_tweet(entry: dict[str, Any]) -> dict[str, Any] | None:
    """The tweet an entry/module item is *about* (not a nested quote/retweet)."""
    for holder in (_path(entry, "content", "itemContent"), _path(entry, "item", "itemContent")):
        r = _unwrap_visibility(_path(holder, "tweet_results", "result") if holder is not None else None)
        if isinstance(r, dict) and looks_like_tweet(r):
            return r
    return None


def _norm(username: str | None) -> str:
    return (username or "").strip().lstrip("@").lower()


def iter_tweets(node: Any) -> Iterator[dict[str, Any]]:
    """Every GraphQL tweet result anywhere in the payload (used to batch-score before pruning)."""
    stack = [node]
    while stack:
        cur = stack.pop()
        if isinstance(cur, dict):
            if looks_like_tweet(cur):
                yield cur
            stack.extend(cur.values())
        elif isinstance(cur, list):
            stack.extend(cur)


# ---------------------------------------------------------------------------------------------------
# Decisions
# ---------------------------------------------------------------------------------------------------

TEXT_SCOPES = ("engagements", "all")


def _mentions(fields: dict[str, Any]) -> list[tuple[str | None, str | None]]:
    """(id_str, screen_name) of every user mention in a legacy/REST tweet dict."""
    out: list[tuple[str | None, str | None]] = []
    for m in _path(fields, "entities", "user_mentions") or []:
        if isinstance(m, dict):
            mid = m.get("id_str") or m.get("id")
            name = m.get("screen_name")
            out.append((str(mid) if mid not in (None, "") else None, name if isinstance(name, str) else None))
    return out


class _Judge:
    """Wraps the decider with owner protection, text scoping and a per-id decision cache."""

    def __init__(self, decider: Any, owner_ids: Iterable[str], owner_usernames: Iterable[str],
                 cache: MutableMapping[str, bool] | None, text_scope: str = "engagements") -> None:
        if text_scope not in TEXT_SCOPES:
            raise ValueError(f"text_scope must be one of {TEXT_SCOPES}, got {text_scope!r}")
        self.decider = decider
        self.owner_ids = {str(i) for i in owner_ids if i}
        self.owner_usernames = {_norm(u) for u in owner_usernames if u}
        self.cache: MutableMapping[str, bool] = cache if cache is not None else {}
        self.text_scope = text_scope
        # author context is passed only to deciders that declare it (stubs and older deciders: unchanged calls)
        self.pass_author = _accepts(getattr(decider, "should_nullify", None), "author")
        score = getattr(decider, "score", None)
        self.pass_authors = callable(score) and _accepts(score, "authors")

    def is_owner(self, uid: str | None, username: str | None) -> bool:
        return (uid is not None and uid in self.owner_ids) or (bool(username) and _norm(username) in self.owner_usernames)

    def blocked(self, uid: str | None, username: str | None) -> bool:
        """Identity-only decision (no text). Prefers the decider's is_blocked() when it has one."""
        if self.is_owner(uid, username) or not (uid or username):
            return False
        key = f"user:{uid or _norm(username)}"
        if key in self.cache:
            return self.cache[key]
        fn = getattr(self.decider, "is_blocked", None)
        verdict = bool(fn(author_id=uid, username=username)) if callable(fn) else bool(
            self.decider.should_nullify("", author_id=uid, username=username))
        self.cache[key] = verdict
        return verdict

    def engages_owner(self, fields: dict[str, Any], quoted_author: tuple[str | None, str | None] | None) -> bool:
        """Does this tweet reply to, mention or quote the owner? `fields` is GraphQL `legacy` or a REST tweet."""
        reply_id = fields.get("in_reply_to_user_id_str")
        reply_name = fields.get("in_reply_to_screen_name")
        if self.is_owner(str(reply_id) if reply_id not in (None, "") else None,
                         reply_name if isinstance(reply_name, str) else None):
            return True
        if any(self.is_owner(mid, name) for mid, name in _mentions(fields)):
            return True
        return quoted_author is not None and self.is_owner(*quoted_author)

    def in_text_scope(self, fields: dict[str, Any], quoted_author: tuple[str | None, str | None] | None) -> bool:
        return self.text_scope == "all" or self.engages_owner(fields, quoted_author)

    def decide(self, tid: str | None, uid: str | None, name: str | None, text: str, in_scope: bool,
               author: dict[str, Any] | None = None) -> bool:
        """Blocklist first (any scope), then the text decider only for in-scope tweets. Cached by tweet id.
        author: the tweet author's context dict, forwarded when the decider accepts it."""
        if self.is_owner(uid, name):
            return False
        key = f"tweet:{tid}" if tid else None
        if key is not None and key in self.cache:
            return self.cache[key]
        if self.blocked(uid, name):
            verdict = True
        elif in_scope:
            if self.pass_author:
                verdict = bool(self.decider.should_nullify(text, author_id=uid, username=name, author=author))
            else:
                verdict = bool(self.decider.should_nullify(text, author_id=uid, username=name))
        else:
            verdict = False
        if key is not None:
            self.cache[key] = verdict
        return verdict

    def tweet(self, t: dict[str, Any]) -> bool:
        uid, name = tweet_author(t)
        return self.decide(tweet_id(t), uid, name, tweet_text(t), self._graphql_in_scope(t),
                           tweet_author_context(t) if self.pass_author else None)

    def _graphql_in_scope(self, t: dict[str, Any]) -> bool:
        if self.text_scope == "all":
            return True
        quoted = _unwrap_visibility(_path(t, "quoted_status_result", "result"))
        quoted_author = tweet_author(quoted) if isinstance(quoted, dict) and looks_like_tweet(quoted) else None
        return self.engages_owner(_dict(t.get("legacy")), quoted_author)

    def warm(self, tweets: list[dict[str, Any]]) -> None:
        """Batch-score uncached, unblocked, in-scope tweet texts once so per-tweet should_nullify() hits the
        classifier's text cache instead of running one forward pass per tweet."""
        score = getattr(self.decider, "score", None)
        if not callable(score):
            return
        texts: list[str] = []
        authors: list[dict[str, Any] | None] = []
        for t in tweets:
            tid = tweet_id(t)
            uid, name = tweet_author(t)
            if (tid and f"tweet:{tid}" in self.cache) or self.is_owner(uid, name):
                continue
            if not self._graphql_in_scope(t):
                continue
            fn = getattr(self.decider, "is_blocked", None)
            if callable(fn) and fn(author_id=uid, username=name):
                continue
            text = tweet_text(t)
            if text:
                texts.append(text)
                authors.append(tweet_author_context(t))
        if texts:
            # same author context as the per-tweet call, so the classifier's cache key matches
            score(texts, authors=authors) if self.pass_authors else score(texts)


# ---------------------------------------------------------------------------------------------------
# GraphQL (generic) pass
# ---------------------------------------------------------------------------------------------------

CURSOR_TYPE = "TimelineTimelineCursor"


def _is_cursor_item(item: Any) -> bool:
    """Module-level cursor items ("Show replies") carry no content of their own.

    Real TweetDetail payloads name them `conversationthread-<id>-cursor-showmore-<n>`, so the entryId
    prefix test used for top-level cursors is not enough.
    """
    if not isinstance(item, dict):
        return False
    eid = item.get("entryId")
    if isinstance(eid, str) and (eid.startswith(CURSOR_PREFIX) or "-cursor-" in eid):
        return True
    for holder in (_path(item, "item", "itemContent"), item.get("itemContent"), _path(item, "item", "content")):
        if isinstance(holder, dict) and CURSOR_TYPE in (holder.get("itemType"), holder.get("__typename"),
                                                         holder.get("entryType")):
            return True
    return False


def _content_items(items: Any) -> list[Any]:
    return [i for i in items if not _is_cursor_item(i)] if isinstance(items, list) else []


class _GraphQLPruner:
    def __init__(self, judge: _Judge) -> None:
        self.judge = judge
        self.report: list[ReportRow] = []

    def walk(self, node: Any, parent_key: str | None = None, in_tweet: bool = False) -> tuple[Any, list[Hit]]:
        """Return (pruned copy of node, nullified hits still present in that copy)."""
        if isinstance(node, dict):
            return self._walk_dict(node, parent_key, in_tweet)
        if isinstance(node, list):
            return self._walk_list(node, in_tweet)
        return node, []

    def _walk_dict(self, d: dict[str, Any], parent_key: str | None, in_tweet: bool) -> tuple[dict[str, Any], list[Hit]]:
        hits: list[Hit] = []
        is_tweet = looks_like_tweet(d)
        if is_tweet and self.judge.tweet(d):
            uid, name = tweet_author(d)
            hits.append(Hit("tweet", tweet_id(d), name, f"tweet {tweet_id(d)} nullified"))
        child_in_tweet = in_tweet or is_tweet
        out: dict[str, Any] = {}
        for k, v in d.items():
            # users nested inside a tweet (author, media source_user, ...) belong to that tweet's decision
            if k == "user_results" and parent_key != "core" and not child_in_tweet:
                uid, name = user_identity(_path(v, "result"))
                if (uid or name) and self.judge.blocked(uid, name):
                    hits.append(Hit("user", uid, name, f"user {uid} blocklisted"))
            out[k], sub = self.walk(v, k, child_in_tweet)
            hits.extend(sub)
        return out, hits

    def _walk_list(self, items: list[Any], in_tweet: bool) -> tuple[list[Any], list[Hit]]:
        out: list[Any] = []
        hits: list[Hit] = []
        for el in items:
            if isinstance(el, dict) and isinstance(el.get("entryId"), str):
                kept = self._entry(el, in_tweet)
                if kept is not None:
                    out.append(kept)
            elif isinstance(el, dict) and isinstance(_path(el, "entry", "entryId"), str):
                kept = self._entry(el["entry"], in_tweet)
                if kept is not None:
                    out.append({**el, "entry": kept})
            else:
                new, sub = self.walk(el, None, in_tweet)
                out.append(new)
                hits.extend(sub)
        return out, hits

    def _entry(self, entry: dict[str, Any], in_tweet: bool = False) -> dict[str, Any] | None:
        """Prune one entry/module item; None means drop it. Hits are consumed here (never bubble past)."""
        entry_id: str = entry["entryId"]
        if entry_id.startswith(CURSOR_PREFIX):
            return entry
        new, hits = self._walk_dict(entry, None, in_tweet)
        old_content = _content_items(_path(entry, "content", "items"))
        new_content = _content_items(_path(new, "content", "items"))
        emptied = bool(old_content) and not new_content
        primary = primary_tweet(entry)
        protected = primary is not None and self.judge.is_owner(*tweet_author(primary))
        if protected:
            return new
        if hits:
            self.report.append({"entryId": entry_id, "username": hits[0].username, "reason": hits[0].reason})
            return None
        if emptied:
            self.report.append({"entryId": entry_id, "username": None, "reason": "all module items nullified"})
            return None
        return new


# ---------------------------------------------------------------------------------------------------
# Legacy REST globalObjects pass (notifications/all.json and friends)
# ---------------------------------------------------------------------------------------------------


def _from_user_id(u: Any) -> str | None:
    """fromUsers entries look like {"user": {"id": "123"}}; tolerate bare ids too."""
    if isinstance(u, dict):
        uid = _path(u, "user", "id") or u.get("id")
    else:
        uid = u
    return str(uid) if uid not in (None, "") else None


def _rest_refs(t: Any) -> list[str]:
    """Tweet ids a REST tweet embeds by reference (its quote and its retweet source)."""
    t = _dict(t)
    return [str(t[k]) for k in ("quoted_status_id_str", "retweeted_status_id_str") if t.get(k) not in (None, "")]


def _notif_user_refs(n: Any) -> list[str]:
    """Every user a REST notification names: aggregate fromUsers plus rich-text message.entities refs."""
    out: list[str] = []
    agg = _dict(_path(n, "template", "aggregateUserActionsV1"))
    for u in agg.get("fromUsers") or []:
        uid = _from_user_id(u)
        if uid:
            out.append(uid)
    for ent in _path(n, "message", "entities") or []:
        uid = _path(ent, "ref", "user", "id")
        if uid not in (None, ""):
            out.append(str(uid))
    return out


def _notif_tweet_refs(n: Any) -> list[str]:
    out: list[str] = []
    agg = _dict(_path(n, "template", "aggregateUserActionsV1"))
    for o in agg.get("targetObjects") or []:
        tid = _path(o, "tweet", "id")
        if tid not in (None, ""):
            out.append(str(tid))
    for ent in _path(n, "message", "entities") or []:
        tid = _path(ent, "ref", "tweet", "id")
        if tid not in (None, ""):
            out.append(str(tid))
    return out


def _prune_global_objects(payload: dict[str, Any], judge: _Judge) -> tuple[dict[str, Any], list[ReportRow], bool]:
    """Returns (payload', report, changed). `changed` can be True with an empty report (e.g. only an
    unreferenced nullified tweet object was removed), so callers must not infer "untouched" from the report."""
    report: list[ReportRow] = []
    go = _dict(payload.get("globalObjects"))
    users: dict[str, Any] = {str(k): v for k, v in _dict(go.get("users")).items()}
    tweets: dict[str, Any] = {str(k): v for k, v in _dict(go.get("tweets")).items()}
    notifs: dict[str, Any] = {str(k): v for k, v in _dict(go.get("notifications")).items()}

    def uname(uid: str | None) -> str | None:
        n = _path(users.get(uid or ""), "screen_name")
        return n if isinstance(n, str) else None

    def author_of(tid: str | None) -> str | None:
        a = _dict(tweets.get(tid or "")).get("user_id_str")
        return str(a) if a not in (None, "") else None

    bad_users = {uid for uid in users if judge.blocked(uid, uname(uid))}

    bad_tweets: dict[str, str] = {}  # tweet id -> reason
    for tid, t in tweets.items():
        t = _dict(t)
        author = author_of(tid)
        quoted = _rest_refs({"quoted_status_id_str": t.get("quoted_status_id_str")})
        quoted_author = (author_of(quoted[0]), uname(author_of(quoted[0]))) if quoted else None
        in_scope = judge.in_text_scope(t, quoted_author)
        ctx = rest_author(users.get(author or "")) if judge.pass_author else None
        if judge.decide(tid, author, uname(author), str(t.get("full_text") or ""), in_scope, ctx):
            bad_tweets[tid] = f"tweet {tid} nullified"
    # Quotes/retweets of a nullified tweet go too (mirrors the GraphQL subtree rule). Iterate to a fixed
    # point so the result does not depend on dict order (RT -> quote -> toxic chains).
    grew = True
    while grew:
        grew = False
        for tid, t in tweets.items():
            if tid in bad_tweets or judge.is_owner(author_of(tid), uname(author_of(tid))):
                continue
            ref = next((r for r in _rest_refs(t) if r in bad_tweets), None)
            if ref is not None:
                bad_tweets[tid] = f"references nullified tweet {ref}"
                grew = True

    # One aggregate policy for REST and GraphQL: a notification that names a blocklisted user or targets a
    # nullified tweet is dropped whole (trimming would leave the name in message.text/entities).
    dropped_notifs: dict[str, tuple[str | None, str]] = {}
    new_notifs: dict[str, Any] = {}
    for nid, n in notifs.items():
        user_refs = _notif_user_refs(n)
        bad_target = next((x for x in _notif_tweet_refs(n) if x in bad_tweets), None)
        bad_user = next((u for u in user_refs if u in bad_users), None)
        if bad_target is not None:
            dropped_notifs[nid] = (uname(user_refs[0]) if user_refs else uname(author_of(bad_target)),
                                   f"targets nullified tweet {bad_target}")
        elif bad_user is not None:
            dropped_notifs[nid] = (uname(bad_user), f"names blocklisted user {bad_user}")
        else:
            new_notifs[nid] = n

    dropped_entry_tweets: set[str] = set()

    def prune_entries(entries: list[Any]) -> list[Any]:
        out: list[Any] = []
        for e in entries:
            eid = _dict(e).get("entryId")
            if not isinstance(eid, str) or eid.startswith(CURSOR_PREFIX):
                out.append(e)
                continue
            content = _dict(_path(e, "content", "item", "content"))
            tid = str(_path(content, "tweet", "id") or "")
            nid = str(_path(content, "notification", "id") or "")
            if tid and tid in bad_tweets:
                report.append({"entryId": eid, "username": uname(author_of(tid)), "reason": bad_tweets[tid]})
                dropped_entry_tweets.add(tid)
                continue
            if nid and nid in dropped_notifs:
                name, reason = dropped_notifs[nid]
                report.append({"entryId": eid, "username": name, "reason": reason})
                continue
            out.append(e)
        return out

    timeline = _dict(payload.get("timeline"))
    new_instructions: list[Any] = []
    for ins in timeline.get("instructions") or []:
        entries = _path(ins, "addEntries", "entries")
        if isinstance(entries, list):
            ins = {**ins, "addEntries": {**ins["addEntries"], "entries": prune_entries(entries)}}
        new_instructions.append(ins)

    # Referential integrity: keep every object a kept tweet or kept notification still points at (owner
    # tweets quoting a nullified tweet keep their quote, as on the GraphQL side), transitively.
    needed_tweets: set[str] = set()
    needed_users: set[str] = set()
    frontier = [tid for tid in tweets if tid not in bad_tweets]
    for n in new_notifs.values():
        frontier.extend(_notif_tweet_refs(n))
        needed_users.update(_notif_user_refs(n))
    while frontier:
        tid = frontier.pop()
        if tid in needed_tweets or tid not in tweets:
            continue
        needed_tweets.add(tid)
        if author_of(tid):
            needed_users.add(author_of(tid) or "")
        frontier.extend(_rest_refs(tweets[tid]))
    removed_tweets = {tid for tid in bad_tweets if tid not in needed_tweets}
    removed_users = {uid for uid in bad_users if uid not in needed_users}

    changed = bool(removed_tweets or removed_users or dropped_notifs or report)
    if not changed:
        return payload, [], False
    for tid in sorted(removed_tweets - dropped_entry_tweets):
        report.append({"entryId": f"globalObjects.tweets.{tid}", "username": uname(author_of(tid)),
                       "reason": bad_tweets[tid]})

    orig_users, orig_tweets = _dict(go.get("users")), _dict(go.get("tweets"))
    new_go = {**go,
              "users": {k: v for k, v in orig_users.items() if str(k) not in removed_users},
              "tweets": {k: v for k, v in orig_tweets.items() if str(k) not in removed_tweets}}
    if "notifications" in go:
        new_go["notifications"] = {k: v for k, v in _dict(go.get("notifications")).items() if str(k) not in dropped_notifs}
    out = {**payload, "globalObjects": new_go}
    if "timeline" in payload:
        out["timeline"] = {**timeline, "instructions": new_instructions}
    return out, report, True


# ---------------------------------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------------------------------


def prune(payload: Any, decider: Any, *, owner_ids: Iterable[str] = (), owner_usernames: Iterable[str] = (),
          cache: MutableMapping[str, bool] | None = None,
          text_scope: str = "engagements") -> tuple[Any, list[ReportRow]]:
    """Strip nullified tweets/users from an x.com JSON payload.

    Returns (new_payload, report). When nothing is pruned, new_payload IS the input object (callers should
    test `new is payload`, not `report`, to decide whether to re-encode). The input is never mutated. On
    any exception the original payload and an empty report are returned.

    owner_ids / owner_usernames: the logged-in account (from the `twid` cookie and GVON_HANDLE). Their
    tweets are never judged and entries whose primary tweet they authored are never dropped.
    cache: optional shared dict of decisions keyed "tweet:<id>" / "user:<id>", reused across responses.
    text_scope: "engagements" (default) sends a tweet's text to the decider only when it replies to,
    mentions or quotes the owner; "all" scores every tweet. The blocklist applies in both scopes.
    """
    try:
        judge = _Judge(decider, owner_ids, owner_usernames, cache, text_scope)
        report: list[ReportRow] = []
        current = payload
        changed = False
        if isinstance(current, dict) and isinstance(current.get("globalObjects"), dict):
            current, rows, changed = _prune_global_objects(current, judge)
            report.extend(rows)
        tweets = list(iter_tweets(current))
        if tweets:
            judge.warm(tweets)
        pruner = _GraphQLPruner(judge)
        new, _ = pruner.walk(current)
        report.extend(pruner.report)
        if not report and not changed:
            return payload, []
        return new, report
    except Exception:  # totality: a pruning bug must never break the page
        log.exception("gvon prune failed; passing payload through untouched")
        return payload, []


# ---------------------------------------------------------------------------------------------------
# CLI: prune a saved JSON file (fixture or data/recordings/*) with the blocklist only, or the real model
# ---------------------------------------------------------------------------------------------------


class BlocklistDecider:
    """Identity-only decider built from data/blocklist.json: no model, no text scoring.

    Used by the CLI (and as the addon's degraded mode when no trained model exists) so recorded payloads
    can be checked against the real blocklist without loading torch.
    """

    def __init__(self, blocklist_path: str | Path) -> None:
        p = Path(blocklist_path)
        data = json.loads(p.read_text() or "{}") if p.exists() else {}
        accts = data.get("accounts") or []
        self.ids = {str(a["id"]) for a in accts if isinstance(a, dict) and a.get("id") not in (None, "")}
        self.usernames = {_norm(a.get("username")) for a in accts if isinstance(a, dict) and a.get("username")}

    def is_blocked(self, *, author_id: str | None = None, username: str | None = None) -> bool:
        return (author_id is not None and str(author_id) in self.ids) or (bool(username) and _norm(username) in self.usernames)

    def score(self, texts: list[str]) -> list[float]:
        return [0.0 for _ in texts]

    def should_nullify(self, text: str, *, author_id: str | None = None, username: str | None = None) -> bool:
        return self.is_blocked(author_id=author_id, username=username)


def main(argv: list[str] | None = None) -> int:
    from gvon.env import ROOT

    ap = argparse.ArgumentParser(description="Prune a saved x.com JSON payload and print the report.")
    ap.add_argument("payload", type=Path, help="JSON file (fixture or data/recordings/*.json)")
    ap.add_argument("--blocklist", type=Path, default=ROOT / "data" / "blocklist.json")
    ap.add_argument("--model", action="store_true", help="use the trained gvon.classifier.Nullifier, not blocklist only")
    ap.add_argument("--owner", action="append", default=[], help="owner username (repeatable)")
    ap.add_argument("--out", type=Path, help="write the pruned JSON here")
    ap.add_argument("--text-scope", choices=TEXT_SCOPES, default="engagements",
                    help="score text only for tweets engaging the owner (default) or for every tweet")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    raw = json.loads(args.payload.read_text())
    # recordings written by the addon wrap the body as {"request": ..., "response": body}
    body = raw["response"] if isinstance(raw, dict) and "response" in raw and "request" in raw else raw
    if args.model:
        from gvon.classifier import Nullifier

        decider: Any = Nullifier(ROOT / "models" / "latest", args.blocklist)
    else:
        decider = BlocklistDecider(args.blocklist)
    new, report = prune(body, decider, owner_usernames=args.owner, text_scope=args.text_scope)
    for row in report:
        print(json.dumps(row))
    print(f"pruned {len(report)} item(s)", file=sys.stderr)
    if args.out:
        args.out.write_text(json.dumps(new))  # ascii escapes: lone UTF-16 surrogates must survive
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
