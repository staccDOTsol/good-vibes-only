"""mitmproxy addon: strip nullified posts out of x.com JSON before it reaches the browser.

Run:  .venv/bin/mitmdump -s gvon/nullifier.py          (port 8080 on ALL interfaces by default: mitmproxy's
      listen_host is empty unless you pass --listen-host 127.0.0.1; see README / docs/NULLIFIER.md)
Setup (CA trust, dedicated proxied browser profile): docs/NULLIFIER.md

Why a MITM proxy: x.com is HTTPS and serves every post from the same hosts, so DNS/hosts blocking can
only block all of X or nothing. A local proxy whose CA the browser trusts can decrypt the timeline JSON,
remove individual entries (gvon.prune) and re-encode the response, so the nullified bytes never reach
the browser's network stack, much like an ad blocker filtering at the request level.

Scope: the x.com / twitter.com web client (browser) only. Native X mobile/desktop apps pin their TLS
certificates and refuse the mitmproxy CA, so their traffic cannot be filtered this way and is left alone.
Nothing leaves the machine: decisions use the local classifier (gvon.classifier.Nullifier) and the
local data/blocklist.json.

Environment options:
  GVON_RECORD=1     dump every matched response body to data/recordings/<ts>_<path>.json (no headers or
                    cookies are written) so real response SHAPES can be studied; recordings are real
                    timeline data and must stay under data/ (fixtures are hand-written synthetic copies)
  GVON_DRY_RUN=1    log what would be pruned but forward responses unmodified
  GVON_THRESHOLD=f  override the model's nullify threshold (resolved in gvon.classifier.resolve_threshold,
                    the same place the classifier CLI uses)
  GVON_TEXT_SCOPE   "engagements" (default): text-score only tweets that reply to / mention / quote you,
                    every other tweet is judged by the blocklist alone; "all": text-score every tweet
  GVON_HANDLE       your username; with the `twid` cookie it identifies the owner (never pruned)
  GVON_MODEL_DIR / GVON_BLOCKLIST  override models/latest and data/blocklist.json

Logging uses Python `logging` (mitmproxy >= 9 routes it to its event log; `ctx.log` is deprecated).
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any
from urllib.parse import unquote

from gvon.env import ROOT, get
from gvon.prune import BlocklistDecider, prune

try:  # tolerant import: `import gvon.nullifier` must work without mitmproxy (tests, CLI replay)
    from mitmproxy import http  # noqa: F401  (type reference only)
except ImportError:  # pragma: no cover
    http = None  # type: ignore[assignment]

log = logging.getLogger("gvon.nullifier")

TEXT_SCOPES = ("engagements", "all")
HOSTS = frozenset({"x.com", "twitter.com", "api.x.com", "api.twitter.com"})
PATH_PREFIXES = ("/i/api/graphql/", "/graphql/", "/i/api/2/notifications/", "/i/api/2/timeline/")
CACHE_MAX = 50_000
RECORD_DIR = ROOT / "data" / "recordings"


def _flag(name: str) -> bool:
    return (get(name) or "").strip().lower() in {"1", "true", "yes", "on"}


def should_process(host: str, path: str, status: int, content_type: str) -> bool:
    """Only successful JSON responses from the X web API hosts/paths are touched."""
    host = (host or "").lower().rstrip(".")
    path = (path or "").split("?", 1)[0]
    return (host in HOSTS and path.startswith(PATH_PREFIXES) and status == 200
            and "json" in (content_type or "").lower())


def owner_id_from_twid(value: str | None) -> str | None:
    """The `twid` cookie is `u=<user id>` (URL-encoded as u%3D<id>); it names the logged-in account."""
    if not value:
        return None
    m = re.fullmatch(r'"?u=(\d+)"?', unquote(value).strip())
    return m.group(1) if m else None


def sanitize_path(path: str, limit: int = 120) -> str:
    """Filesystem-safe name from the request path (query dropped; it can be long and is in the body)."""
    p = path.split("?", 1)[0].strip("/")
    return (re.sub(r"[^A-Za-z0-9._-]+", "_", p) or "root")[:limit]


class BoundedCache(OrderedDict):
    """Decision cache keyed by tweet/user id; evicts oldest entries past CACHE_MAX."""

    def __init__(self, maxlen: int = CACHE_MAX) -> None:
        super().__init__()
        self.maxlen = maxlen

    def __setitem__(self, key: Any, value: Any) -> None:
        super().__setitem__(key, value)
        self.move_to_end(key)
        while len(self) > self.maxlen:
            self.popitem(last=False)


def text_scope_from_env() -> str:
    """GVON_TEXT_SCOPE, validated; an unknown value falls back to the safe default (engagements)."""
    scope = (get("GVON_TEXT_SCOPE") or "engagements").strip().lower()
    if scope not in TEXT_SCOPES:
        log.warning("gvon: unknown GVON_TEXT_SCOPE=%r; using 'engagements'", scope)
        return "engagements"
    return scope


def build_decider(threshold: float | None = None) -> Any:
    """Build the classifier once. If no trained model exists yet, fall back LOUDLY to blocklist-only
    filtering (data/blocklist.json) rather than refusing to proxy. Logged at WARNING: mitmdump exits
    on any ERROR logged during start-up."""
    model_dir = Path(get("GVON_MODEL_DIR") or ROOT / "models" / "latest")
    blocklist = Path(get("GVON_BLOCKLIST") or ROOT / "data" / "blocklist.json")
    t0 = time.perf_counter()
    try:
        from gvon.classifier import Nullifier  # lazy: torch is heavy and may be absent in tests

        nul = Nullifier(model_dir, blocklist, threshold=threshold, platform="x")
        warm = getattr(nul, "warmup", None)
        if callable(warm):
            warm()
        log.info("gvon: Nullifier loaded from %s on device=%s threshold=%s (source=%s, text_model=%s) in %.2fs",
                 model_dir, getattr(nul, "device", "unknown"), getattr(nul, "threshold", threshold),
                 getattr(nul, "threshold_source", "?"), getattr(nul, "text_model_enabled", "?"),
                 time.perf_counter() - t0)
        return nul
    except Exception as exc:
        log.warning("gvon: DEGRADED MODE, classifier unavailable (%s: %s); filtering by blocklist %s only",
                  type(exc).__name__, exc, blocklist)
        return BlocklistDecider(blocklist)


class GvonNullifier:
    """The mitmproxy addon. Construction is cheap; the classifier is built in load()."""

    def __init__(self, decider: Any | None = None) -> None:
        self.decider = decider
        self.cache: BoundedCache = BoundedCache()
        self.record = _flag("GVON_RECORD")
        self.dry_run = _flag("GVON_DRY_RUN")
        self.text_scope = text_scope_from_env()
        handle = (get("GVON_HANDLE") or "").strip().lstrip("@")
        self.owner_usernames = {handle} if handle else set()
        self._blocklist_mtime: float | None = None

    # mitmproxy lifecycle -------------------------------------------------------------------------
    def load(self, loader: Any) -> None:
        if self.decider is None:
            self.decider = build_decider()
        log.info("gvon: nullifier addon ready (dry_run=%s record=%s owner=%s text_scope=%s)",
                 self.dry_run, self.record, ",".join(sorted(self.owner_usernames)) or "unset", self.text_scope)

    def response(self, flow: Any) -> None:
        try:
            self._handle(flow)
        except Exception:  # never break browsing because of the filter
            log.exception("gvon: response hook failed; forwarding unmodified")

    # internals -----------------------------------------------------------------------------------
    def _maybe_reload_blocklist(self) -> None:
        """Pick up a regenerated data/blocklist.json without restarting the proxy."""
        reload = getattr(self.decider, "reload_blocklist", None)
        path = getattr(self.decider, "blocklist_path", None)
        if not callable(reload) or path is None:
            return
        try:
            mtime = Path(path).stat().st_mtime
        except OSError:
            return
        if self._blocklist_mtime is None:
            self._blocklist_mtime = mtime
        elif mtime != self._blocklist_mtime:
            self._blocklist_mtime = mtime
            reload()
            self.cache.clear()
            log.info("gvon: blocklist changed on disk; reloaded and cleared decision cache")

    def _handle(self, flow: Any) -> None:
        req, resp = flow.request, flow.response
        if resp is None or not should_process(req.pretty_host, req.path, resp.status_code,
                                              resp.headers.get("content-type", "")):
            return
        try:
            body = resp.json()
        except Exception:
            return
        if self.record:
            self._record(req, body)
        if self.decider is None:
            self.decider = build_decider()
        self._maybe_reload_blocklist()

        owner_ids = {i for i in [owner_id_from_twid(req.cookies.get("twid"))] if i}
        new, report = prune(body, self.decider, owner_ids=owner_ids, owner_usernames=self.owner_usernames,
                            cache=self.cache, text_scope=self.text_scope)
        if new is body:  # prune() returns the input object itself when nothing changed
            return
        op = req.path.split("?", 1)[0].rsplit("/", 1)[-1]
        for row in report:
            log.info("gvon: %s %s entry=%s @%s (%s)", "WOULD PRUNE" if self.dry_run else "pruned", op,
                     row.get("entryId"), row.get("username") or "?", row.get("reason"))
        if not self.dry_run:
            # ensure_ascii=True (the default) on purpose: X sometimes emits unpaired UTF-16 surrogates at
            # truncation boundaries; as raw chars they cannot be UTF-8 encoded (set_text would raise and the
            # page would go out unfiltered), while the \uXXXX escape round-trips through JSON.parse.
            resp.text = json.dumps(new, separators=(",", ":"))

    def _record(self, req: Any, body: Any) -> None:
        """Save the response body (plus method/host/path, never headers or cookies) under data/."""
        try:
            RECORD_DIR.mkdir(parents=True, exist_ok=True)
            ts = time.strftime("%Y%m%dT%H%M%S") + f"{time.time() % 1:.3f}"[1:]
            out = RECORD_DIR / f"{ts}_{sanitize_path(req.path)}.json"
            out.write_text(json.dumps({"request": {"method": req.method, "host": req.pretty_host,
                                                   "path": req.path.split("?", 1)[0]},
                                       "response": body}))  # ascii escapes: lone surrogates survive
        except Exception:
            log.exception("gvon: failed to record response")


addons = [GvonNullifier()]


def main(argv: list[str] | None = None) -> int:
    """Replay a recorded/fixture body through the same decider the proxy would build (no network)."""
    ap = argparse.ArgumentParser(description="GVON mitmproxy addon. Run the proxy with "
                                             "`mitmdump -s gvon/nullifier.py`; this CLI replays a saved body.")
    ap.add_argument("payload", type=Path, help="JSON body or data/recordings/*.json file")
    ap.add_argument("--owner-id", action="append", default=[])
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    raw = json.loads(args.payload.read_text())
    body = raw["response"] if isinstance(raw, dict) and {"request", "response"} <= raw.keys() else raw
    addon = GvonNullifier()
    addon.decider = build_decider()
    _, report = prune(body, addon.decider, owner_ids=args.owner_id, owner_usernames=addon.owner_usernames,
                      text_scope=addon.text_scope)
    for row in report:
        print(json.dumps(row))
    print(f"pruned {len(report)} item(s)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
