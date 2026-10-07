"""Local, fast "student" classifier: sentence embeddings + a logistic-regression head.

Why this shape: a frontier model labels engagements offline (expensive, slow, high quality); this
module distils those labels into something that can sit in the network path of x.com responses.
MiniLM-L6 embeddings (~22M params) + a linear head run in milliseconds on CPU or Apple Metal (MPS),
so the proxy can score a timeline page without noticeable latency.

Public API (see CONTRACT.md):
    Nullifier(model_dir, blocklist_path, threshold, watchlist_path).score(texts, authors=None) -> list[float]
    Nullifier(...).should_nullify(text, author_id=..., username=..., author=None) -> bool

Inputs (config.json `student_inputs`):
- v1 ["tweet_text"]: the post text only, cleaned by content_text() (leading @mention run and URLs removed,
  exactly as in gvon.train).
- v2 ["tweet_text", "author_bio", "author_stats"]: the same text embedding, plus a MiniLM embedding of the
  author's bio (zeros when there is none) and a scalar block built by author_scalars() from the optional
  author dict {bio, followers, following, tweet_count, created_at, verified}: log1p counts, log1p account
  age in days, log1p(followers / (following + 1)), verified, has_bio, standardised by the StandardScaler in
  models/latest/author_scaler.joblib, plus one 0/1 missing-indicator column per field that can be unknown.
  A missing author (None) or missing field is never NaN: the standardised value is 0 (the training mean)
  and its indicator is 1, exactly as rows without author context were seen in training.
  This is the cheap author context the teacher saw; it does not include the parent tweet or the author's
  other posts.
Texts with < MIN_CONTENT_CHARS of content (link-only / media-only replies) score 0.0 whatever the author:
the model abstains and leaves them to the blocklist.

Threshold: one resolution order everywhere (proxy, CLI, tests): explicit argument > GVON_THRESHOLD >
config.json > 0.5 (resolve_threshold). If training found no threshold meeting its precision target it
writes text_model_enabled=false; text scoring is then off unless a threshold is given explicitly.

Per-author signals: accounts in data/watchlist.json (teacher verdict "watch") get a lower threshold
(threshold - GVON_WATCH_DELTA, default 0.15), and a bounded running mean of each author's distinct scored
texts nullifies an author once n >= AUTHOR_AGG_MIN_N and mean P >= threshold.

CLI:
    python -m gvon.classifier "text" ["text2" ...]   # print P(nullify) per text
    python -m gvon.classifier --bench                # time load + a 50-text batch, print device + ms
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sys
import threading
import time
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from gvon import env

log = logging.getLogger("gvon.classifier")

# Anchored at the repo root (where .env lives), not the cwd, so running from another directory never
# reads or writes real data outside this repo's gitignored data/ and models/.
DEFAULT_MODEL_DIR = env.ROOT / "models" / "latest"
DEFAULT_BLOCKLIST = env.ROOT / "data" / "blocklist.json"
WATCHLIST_NAME = "watchlist.json"  # written by gvon.label next to blocklist.json
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CACHE_MAX = 100_000
ENCODE_BATCH = 64
MIN_CONTENT_CHARS = 3
DEFAULT_WATCH_DELTA = 0.15
AUTHOR_AGG_MIN_N = 3
AUTHOR_AGG_MAX_AUTHORS = 20_000
AUTHOR_AGG_MAX_TEXTS = 50

BIO_CACHE_MAX = 20_000

# student_inputs values written to config.json by gvon.train
INPUT_TEXT = "tweet_text"
INPUT_BIO = "author_bio"
INPUT_STATS = "author_stats"
STUDENT_INPUTS_TEXT = [INPUT_TEXT]
STUDENT_INPUTS_AUTHOR = [INPUT_TEXT, INPUT_BIO, INPUT_STATS]
AUTHOR_SCALER_FILE = "author_scaler.joblib"
# Scalar block, in column order. The first len(AUTHOR_SCALAR_NAMES) columns are standardised by the saved
# StandardScaler (NaN = unknown, ignored when fitting, 0 after scaling); the indicator columns stay 0/1.
AUTHOR_SCALAR_NAMES = ("log1p_followers", "log1p_following", "log1p_tweet_count", "log1p_account_age_days",
                       "log1p_followers_per_following", "verified", "has_bio")
AUTHOR_MISSING_NAMES = ("followers_missing", "following_missing", "tweet_count_missing", "account_age_missing",
                        "verified_missing")
AUTHOR_FIELDS = ("bio", "followers", "following", "tweet_count", "created_at", "verified")

_LEADING_MENTIONS = re.compile(r"^(?:\s*@\w+)+")
_URL = re.compile(r"https?://\S+")


def content_text(text: str | None) -> str:
    """What the student actually reads: the post minus its leading @mention run and URLs.

    Reply text from the X API starts with the @handles being replied to (nearly always the owner's), a
    reply-target cue the home timeline lacks; t.co links carry no meaning. Shared by train and score so
    both sides see identical input.
    """
    t = _LEADING_MENTIONS.sub("", text or "")
    return " ".join(_URL.sub(" ", t).split())


def is_low_info(text: str | None) -> bool:
    """Link-only / media-only / mention-only posts: nothing for a text model to judge."""
    return len(content_text(text)) < MIN_CONTENT_CHARS


# ------------------------------------------------------------------------------------------ author context

def parse_created_at(value: Any) -> datetime | None:
    """ISO 8601 (X API v2, Telegram) or the legacy X format "Wed Oct 10 20:19:24 +0000 2018" -> aware UTC."""
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if not isinstance(value, str) or not value.strip():
        return None
    v = value.strip()
    try:
        dt = datetime.fromisoformat(v.replace("Z", "+00:00"))
    except ValueError:
        try:
            dt = datetime.strptime(v, "%a %b %d %H:%M:%S %z %Y")
        except ValueError:
            return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _count(value: Any) -> float | None:
    """A non-negative finite count, or None (unknown). Booleans and junk are unknown, not 0/1."""
    if isinstance(value, bool) or value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) and v >= 0 else None


def _flag(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)) and value in (0, 1):
        return float(value)
    return None


def author_bio(author: dict[str, Any] | None) -> str:
    """The bio exactly as the bio encoder sees it (whitespace collapsed); "" when absent or unknown."""
    bio = (author or {}).get("bio")
    return " ".join(bio.split()) if isinstance(bio, str) else ""


def author_scalars(author: dict[str, Any] | None, now: datetime | None = None) -> np.ndarray:
    """Unscaled scalar block for one author: AUTHOR_SCALAR_NAMES (NaN = unknown) ++ AUTHOR_MISSING_NAMES (0/1).

    author: None or a dict with any of AUTHOR_FIELDS. now: the moment the post was written (training) or
    scored (inference); account age = now - created_at in days, clamped at 0."""
    a = author or {}
    followers, following, tweets = _count(a.get("followers")), _count(a.get("following")), _count(a.get("tweet_count"))
    created = parse_created_at(a.get("created_at"))
    ref = now or datetime.now(timezone.utc)
    if ref.tzinfo is None:
        ref = ref.replace(tzinfo=timezone.utc)
    age = max((ref - created).total_seconds() / 86400.0, 0.0) if created is not None else None
    verified = _flag(a.get("verified"))
    nan = float("nan")

    def lg(v: float | None) -> float:
        return float(np.log1p(v)) if v is not None else nan

    ratio = (followers / (following + 1.0)) if followers is not None and following is not None else None
    return np.asarray([
        lg(followers), lg(following), lg(tweets), lg(age), lg(ratio),
        verified if verified is not None else nan,
        1.0 if author_bio(a) else 0.0,
        float(followers is None), float(following is None), float(tweets is None), float(age is None),
        float(verified is None),
    ], dtype=np.float64)


def author_scalar_matrix(authors: list[dict[str, Any] | None], nows: list[datetime | None] | None = None) -> np.ndarray:
    """(n, len(AUTHOR_SCALAR_NAMES) + len(AUTHOR_MISSING_NAMES)) unscaled matrix, NaN where unknown."""
    width = len(AUTHOR_SCALAR_NAMES) + len(AUTHOR_MISSING_NAMES)
    if not authors:
        return np.zeros((0, width), dtype=np.float64)
    nows = nows if nows is not None else [None] * len(authors)
    return np.vstack([author_scalars(a, t) for a, t in zip(authors, nows)])


def scale_author_block(raw: np.ndarray, scaler: Any) -> np.ndarray:
    """Standardise the scalar columns with the fitted scaler, then unknown -> 0 (the training mean); the
    missing-indicator columns pass through. The result never contains NaN."""
    k = len(AUTHOR_SCALAR_NAMES)
    if raw.shape[0] == 0:
        return raw.astype(np.float32)
    import warnings

    with warnings.catch_warnings():  # an all-unknown column (e.g. Telegram only) is legal
        warnings.simplefilter("ignore", RuntimeWarning)
        scaled = np.asarray(scaler.transform(raw[:, :k]), dtype=np.float64)
    scaled = np.where(np.isfinite(scaled), scaled, 0.0)
    return np.hstack([scaled, raw[:, k:]]).astype(np.float32)


def fit_author_scaler(raw: np.ndarray) -> Any:
    """StandardScaler over the scalar columns; NaN (unknown) is ignored when fitting."""
    import warnings

    from sklearn.preprocessing import StandardScaler

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return StandardScaler().fit(raw[:, : len(AUTHOR_SCALAR_NAMES)])


def author_signature(author: dict[str, Any] | None, now: datetime | None = None) -> str:
    """Cache key part for the author context: sha1 of the bio and the (day-rounded) scalar block."""
    vals = author_scalars(author, now)
    sig = author_bio(author) + "\x00" + ",".join("n" if not np.isfinite(v) else f"{v:.3f}" for v in vals)
    return text_key(sig)


def author_from_x_user(row: dict[str, Any] | None) -> dict[str, Any] | None:
    """data/raw/users.jsonl row (X API v2 user) -> author dict."""
    if not isinstance(row, dict):
        return None
    pm = row.get("public_metrics") if isinstance(row.get("public_metrics"), dict) else {}
    return {"bio": row.get("description"), "followers": pm.get("followers_count"),
            "following": pm.get("following_count"), "tweet_count": pm.get("tweet_count"),
            "created_at": row.get("created_at"), "verified": row.get("verified")}


def author_from_telegram_user(row: dict[str, Any] | None) -> dict[str, Any] | None:
    """data/raw/telegram_users.jsonl row -> author dict. Telegram has no public follower counts or account
    creation date, so those stay unknown; description is null when the bio was not fetched."""
    if not isinstance(row, dict):
        return None
    return {"bio": row.get("description"), "followers": None, "following": None, "tweet_count": None,
            "created_at": None, "verified": row.get("verified")}


def resolve_threshold(explicit: float | None, config: dict[str, Any]) -> tuple[float, str]:
    """(threshold, source) in the one order every entry point uses: arg > GVON_THRESHOLD > config > 0.5."""
    if explicit is not None:
        return float(explicit), "argument"
    env_thr = (env.get("GVON_THRESHOLD") or "").strip()
    if env_thr:
        return float(env_thr), "env"
    if config.get("threshold") is not None:
        return float(config["threshold"]), "config"
    return 0.5, "default"


def pick_device() -> str:
    """Return the torch device to embed on.

    GVON_DEVICE (env or .env) wins so a user can force "cpu" when MPS is slower for tiny batches
    or misbehaves; otherwise prefer Apple Metal when present.
    """
    forced = env.get("GVON_DEVICE")
    if forced:
        return forced.strip()
    import torch

    return "mps" if torch.backends.mps.is_available() else "cpu"


def load_encoder(model_name: str, device: str) -> Any:
    """Load a SentenceTransformer, imported lazily because importing torch/transformers costs seconds.

    Tries the local Hugging Face cache first (local_files_only) so a warm start makes no network
    round-trips (those cost ~2s); only the very first run falls through to a download.
    """
    from sentence_transformers import SentenceTransformer

    try:
        from transformers.utils import logging as hf_logging

        hf_logging.disable_progress_bar()
    except Exception:  # cosmetic only
        pass
    try:
        return SentenceTransformer(model_name, device=device, local_files_only=True)
    except Exception:
        return SentenceTransformer(model_name, device=device)


def encode(encoder: Any, texts: list[str], batch_size: int = ENCODE_BATCH) -> np.ndarray:
    """Embed texts as L2-normalised float32 vectors (normalisation keeps the linear head scale-free)."""
    if not texts:
        dim = encoder.get_sentence_embedding_dimension() or 0
        return np.zeros((0, dim), dtype=np.float32)
    vecs = encoder.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return np.asarray(vecs, dtype=np.float32)


class CalibratedHead:
    """Linear head + Platt calibrator, pickled as models/latest/head.joblib by gvon.train.

    Why: the head is trained with class-balanced sample weights, so its raw predict_proba is not a real
    P(nullify) and its scale moves with every retrain. The calibrator is a 1-D logistic fit (unweighted)
    on out-of-fold decision values, so predict_proba estimates the real nullify rate and a threshold such
    as 0.6 keeps its meaning across retrains.
    """

    def __init__(self, base: Any, calibrator: Any) -> None:
        self.base = base
        self.calibrator = calibrator
        self.classes_ = getattr(calibrator, "classes_", getattr(base, "classes_", np.asarray([0, 1])))

    def decision_function(self, x: np.ndarray) -> np.ndarray:
        return self.base.decision_function(x)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return self.calibrator.predict_proba(np.asarray(self.base.decision_function(x)).reshape(-1, 1))


def text_key(text: str) -> str:
    """Cache key: sha1 of the UTF-8 text, so the cache does not hold raw tweet text as keys."""
    return hashlib.sha1(text.encode("utf-8", "surrogatepass")).hexdigest()


def normalize_username(username: str | None) -> str:
    """Usernames compare case-insensitively and without a leading '@'."""
    return (username or "").strip().lstrip("@").strip().lower()


DEFAULT_PLATFORM = "x"  # accounts written before blocklists carried a "platform" key are X accounts


def normalize_platform(platform: str | None) -> str:
    return (platform or DEFAULT_PLATFORM).strip().lower() or DEFAULT_PLATFORM


def load_blocklist_by_platform(path: str | Path) -> dict[str, tuple[set[str], set[str]]]:
    """Read data/blocklist.json (or watchlist.json) into {platform: (ids, usernames)}.

    A missing file means "nobody listed", not an error."""
    p = Path(path)
    if not p.exists():
        return {}
    data = json.loads(p.read_text() or "{}")
    out: dict[str, tuple[set[str], set[str]]] = {}
    for acct in data.get("accounts", []) or []:
        ids, names = out.setdefault(normalize_platform(acct.get("platform")), (set(), set()))
        if acct.get("id") not in (None, ""):
            ids.add(str(acct["id"]).strip())
        name = normalize_username(acct.get("username"))
        if name:
            names.add(name)
    return out


def merge_platforms(by_platform: dict[str, tuple[set[str], set[str]]], platform: str | None = None) -> tuple[set[str], set[str]]:
    """(ids, usernames) for one platform, or the union of all platforms when platform is None."""
    if platform is not None:
        ids, names = by_platform.get(normalize_platform(platform), (set(), set()))
        return set(ids), set(names)
    ids: set[str] = set()
    names: set[str] = set()
    for i, n in by_platform.values():
        ids |= i
        names |= n
    return ids, names


def load_blocklist(path: str | Path, platform: str | None = None) -> tuple[set[str], set[str]]:
    """Read data/blocklist.json (or watchlist.json, same shape) into (ids, usernames).

    platform=None (default) returns every platform's accounts; "x" / "telegram" only that platform's
    (accounts without a "platform" key count as "x"). A missing file means "nobody listed"."""
    return merge_platforms(load_blocklist_by_platform(path), platform)


def read_config(model_dir: str | Path) -> dict[str, Any]:
    """Load models/<dir>/config.json written by gvon.train."""
    cfg_path = Path(model_dir) / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"{cfg_path} not found; run `python -m gvon.train` first")
    return json.loads(cfg_path.read_text())


class Nullifier:
    """Scores text for P(nullify) and decides whether to hide it.

    The embedding model is loaded lazily on the first score() so constructing a Nullifier (e.g. at
    proxy start-up) is cheap and blocklist-only decisions never pay for torch.
    """

    def __init__(
        self,
        model_dir: str | Path = DEFAULT_MODEL_DIR,
        blocklist_path: str | Path = DEFAULT_BLOCKLIST,
        threshold: float | None = None,
        watchlist_path: str | Path | None = None,
        platform: str | None = None,
    ) -> None:
        import joblib

        # Default platform scope for is_blocked/is_watched when a caller passes platform=None: the x.com
        # proxy constructs with platform="x" so a blocked Telegram @name never hides a same-named X account.
        self.default_platform = platform

        self.model_dir = Path(model_dir)
        self.config = read_config(self.model_dir)
        self.head = joblib.load(self.model_dir / "head.joblib")
        self.embedding_model: str = self.config.get("embedding_model", DEFAULT_EMBEDDING_MODEL)
        self.threshold, self.threshold_source = resolve_threshold(threshold, self.config)
        # An explicit threshold (argument or GVON_THRESHOLD) is the user's override and re-enables text
        # scoring even when training found no threshold meeting its precision target.
        self.text_model_enabled: bool = (self.threshold_source in ("argument", "env")
                                         or bool(self.config.get("text_model_enabled", True)))
        self.watch_delta = float((env.get("GVON_WATCH_DELTA") or "").strip() or DEFAULT_WATCH_DELTA)
        self.blocklist_path = Path(blocklist_path)
        self.watchlist_path = Path(watchlist_path) if watchlist_path is not None else self.blocklist_path.parent / WATCHLIST_NAME
        self._load_lists()
        self.device: str = pick_device()
        self._encoder: Any = None
        self._cache: OrderedDict[str, float] = OrderedDict()
        self._authors: OrderedDict[str, OrderedDict[str, float]] = OrderedDict()
        self._bio_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._lock = threading.Lock()
        self._pos_col = self._positive_column()
        self.student_inputs: list[str] = list(self.config.get("student_inputs") or STUDENT_INPUTS_TEXT)
        unknown = set(self.student_inputs) - set(STUDENT_INPUTS_AUTHOR)
        if unknown:
            raise ValueError(f"{self.model_dir}: unsupported student_inputs {sorted(unknown)}")
        # True when the head expects the author block (v2): callers that can supply author context should
        self.uses_author: bool = INPUT_STATS in self.student_inputs or INPUT_BIO in self.student_inputs
        self.uses_author_bio: bool = INPUT_BIO in self.student_inputs
        self.author_scaler: Any = None
        if INPUT_STATS in self.student_inputs:
            self.author_scaler = joblib.load(self.model_dir / AUTHOR_SCALER_FILE)

    def _positive_column(self) -> int:
        """Column of predict_proba for label 1 (nullify); guards against a head trained with odd classes_."""
        classes = list(getattr(self.head, "classes_", [0, 1]))
        return classes.index(1) if 1 in classes else len(classes) - 1

    def _load_lists(self) -> None:
        self._blocked_by_platform = load_blocklist_by_platform(self.blocklist_path)
        self._watch_by_platform = load_blocklist_by_platform(self.watchlist_path)
        # every platform together: what callers that pass no platform match against
        self.blocked_ids, self.blocked_usernames = merge_platforms(self._blocked_by_platform)
        self.watch_ids, self.watch_usernames = merge_platforms(self._watch_by_platform)

    def reload_blocklist(self) -> None:
        """Re-read the block/watch lists (the labeler may regenerate them while the proxy runs)."""
        self._load_lists()

    @staticmethod
    def _listed(ids: set[str], names: set[str], author_id: str | None, username: str | None) -> bool:
        if author_id not in (None, "") and str(author_id).strip() in ids:
            return True
        name = normalize_username(username)
        return bool(name) and name in names

    def _ensure_encoder(self) -> Any:
        if self._encoder is None:
            self._encoder = load_encoder(self.embedding_model, self.device)
        return self._encoder

    def warmup(self) -> None:
        """Load the encoder and run one tiny batch so the first real request is not slow."""
        with self._lock:
            encode(self._ensure_encoder(), ["warmup"])

    def _cache_put(self, key: str, value: float) -> None:
        self._cache[key] = value
        self._cache.move_to_end(key)
        while len(self._cache) > CACHE_MAX:
            self._cache.popitem(last=False)

    def score(self, texts: list[str], authors: list[dict[str, Any] | None] | None = None) -> list[float]:
        """P(nullify) for each text (0.0 = abstain for low-info texts). Batched; one forward pass embeds
        every uncached text (and, for a v2 model, every uncached bio).

        authors: optional list aligned with texts, each None or {bio, followers, following, tweet_count,
        created_at, verified}. Ignored by a text-only (v1) model. For a v2 model a missing author is scored
        with every author feature unknown. Cache key: sha1 of the cleaned text (+ the author signature for v2).
        """
        cleaned = [content_text(str(t)) for t in texts]
        if authors is not None and len(authors) != len(texts):
            raise ValueError(f"authors must align with texts ({len(authors)} != {len(texts)})")
        auth: list[dict[str, Any] | None] = list(authors) if authors is not None else [None] * len(texts)
        now = datetime.now(timezone.utc)
        if self.uses_author:
            keys = [text_key(c) + ":" + author_signature(a, now) for c, a in zip(cleaned, auth)]
        else:
            keys = [text_key(c) for c in cleaned]
        with self._lock:
            missing: dict[str, int] = {}
            for i, (k, c) in enumerate(zip(keys, cleaned)):
                if k in self._cache:
                    self._cache.move_to_end(k)
                elif len(c) < MIN_CONTENT_CHARS:
                    self._cache_put(k, 0.0)
                elif k not in missing:
                    missing[k] = i
            if missing:
                idx = list(missing.values())
                x = self._features([cleaned[i] for i in idx], [auth[i] for i in idx], now)
                probs = self.head.predict_proba(x)[:, self._pos_col]
                for k, p in zip(missing.keys(), probs):
                    self._cache_put(k, float(p))
            return [self._cache[k] for k in keys]

    def _features(self, cleaned: list[str], authors: list[dict[str, Any] | None], now: datetime) -> np.ndarray:
        """Feature rows in the training layout (see gvon.train.build_features). Caller holds the lock."""
        if not self.uses_author:
            return encode(self._ensure_encoder(), cleaned)
        bios = [author_bio(a) if self.uses_author_bio else "" for a in authors]
        new_bios = list(dict.fromkeys(b for b in bios if b and text_key(b) not in self._bio_cache))
        vecs = encode(self._ensure_encoder(), cleaned + new_bios)  # one forward pass for texts and bios
        text_vecs = vecs[: len(cleaned)]
        for b, v in zip(new_bios, vecs[len(cleaned):]):
            self._bio_cache[text_key(b)] = v
            while len(self._bio_cache) > BIO_CACHE_MAX:
                self._bio_cache.popitem(last=False)
        parts = [text_vecs]
        if self.uses_author_bio:
            zero = np.zeros(text_vecs.shape[1], dtype=np.float32)
            parts.append(np.vstack([self._bio_cache[text_key(b)] if b else zero for b in bios]).astype(np.float32))
        if self.author_scaler is not None:
            parts.append(scale_author_block(author_scalar_matrix(authors, [now] * len(authors)), self.author_scaler))
        return np.hstack(parts).astype(np.float32)

    def is_blocked(self, *, author_id: str | None = None, username: str | None = None,
                   platform: str | None = None) -> bool:
        """True when the author is on the blocklist by id or (case-insensitive, '@'-stripped) username.
        platform=None falls back to the constructor's platform; None there matches any platform."""
        platform = platform if platform is not None else self.default_platform
        if platform is None:
            return self._listed(self.blocked_ids, self.blocked_usernames, author_id, username)
        return self._listed(*merge_platforms(self._blocked_by_platform, platform), author_id, username)

    def is_watched(self, *, author_id: str | None = None, username: str | None = None,
                   platform: str | None = None) -> bool:
        """True for "watch" accounts (data/watchlist.json); platform as in is_blocked."""
        platform = platform if platform is not None else self.default_platform
        if platform is None:
            return self._listed(self.watch_ids, self.watch_usernames, author_id, username)
        return self._listed(*merge_platforms(self._watch_by_platform, platform), author_id, username)

    def threshold_for(self, *, author_id: str | None = None, username: str | None = None,
                      platform: str | None = None) -> float:
        """Watch-listed authors get a lower bar (threshold - GVON_WATCH_DELTA)."""
        if self.is_watched(author_id=author_id, username=username, platform=platform):
            return self.threshold - self.watch_delta
        return self.threshold

    def _author_mean(self, author: str, text: str, p: float) -> tuple[float, int]:
        """Record p for this author's distinct text; return (mean P, n distinct texts). Bounded LRU."""
        with self._lock:
            texts = self._authors.get(author)
            if texts is None:
                texts = self._authors[author] = OrderedDict()
            self._authors.move_to_end(author)
            texts[text_key(content_text(text))] = p
            texts.move_to_end(text_key(content_text(text)))
            while len(texts) > AUTHOR_AGG_MAX_TEXTS:
                texts.popitem(last=False)
            while len(self._authors) > AUTHOR_AGG_MAX_AUTHORS:
                self._authors.popitem(last=False)
            return float(sum(texts.values()) / len(texts)), len(texts)

    def should_nullify(self, text: str, *, author_id: str | None = None, username: str | None = None,
                       platform: str | None = None, author: dict[str, Any] | None = None) -> bool:
        """Blocklist first (no model call), then the text model: score >= the author's threshold, or the
        author's running mean over >= AUTHOR_AGG_MIN_N distinct texts >= threshold. Low-info texts abstain.
        platform (None = any) scopes the block/watch lookups and the per-author running mean.
        author: optional {bio, followers, following, tweet_count, created_at, verified} for a v2 model."""
        if self.is_blocked(author_id=author_id, username=username, platform=platform):
            return True
        if not self.text_model_enabled or is_low_info(text):
            return False
        p = (self.score([text]) if author is None else self.score([text], authors=[author]))[0]
        author = str(author_id).strip() if author_id not in (None, "") else normalize_username(username)
        if author and platform is not None:
            author = f"{normalize_platform(platform)}:{author}"
        # record every scored post (including ones that cross the threshold alone) before deciding
        mean, n = self._author_mean(author, text, p) if author else (p, 1)
        if p >= self.threshold_for(author_id=author_id, username=username, platform=platform):
            return True
        if n >= AUTHOR_AGG_MIN_N and mean >= self.threshold:
            log.info("gvon: author aggregate: %s mean P=%.3f over %d posts >= %.3f", username or author, mean, n,
                     self.threshold)
            return True
        return False


def bench_texts(n: int = 50) -> list[str]:
    """Synthetic, clearly-fake benchmark strings (unique so the cache cannot short-circuit the timing)."""
    return [f"benchmark sentence number {i} about nothing in particular, run {time.time_ns()}" for i in range(n)]


def bench_authors(n: int = 50) -> list[dict[str, Any]]:
    """Synthetic author dicts with unique bios (the worst case: every bio is a cache miss)."""
    return [{"bio": f"synthetic benchmark bio {i} {time.time_ns()}", "followers": 10 * i, "following": 100 + i,
             "tweet_count": 1000 + i, "created_at": "2020-01-01T00:00:00Z", "verified": bool(i % 2)} for i in range(n)]


def run_bench(model_dir: str | Path, blocklist_path: str | Path) -> dict[str, Any]:
    """Time: construct + encoder load + first encode, then a cold 50-text batch, then a cached repeat.
    For a model with author inputs every text carries a synthetic author with a unique bio."""
    t0 = time.perf_counter()
    nul = Nullifier(model_dir, blocklist_path)
    t_init = time.perf_counter()
    nul.warmup()
    t_load = time.perf_counter()
    texts = bench_texts(50)
    authors = bench_authors(50) if nul.uses_author else None
    nul.score(texts, authors)
    t_batch = time.perf_counter()
    nul.score(texts, authors)
    t_cached = time.perf_counter()
    texts2 = bench_texts(50)
    authors2 = bench_authors(50) if nul.uses_author else None
    t2 = time.perf_counter()
    nul.score(texts2, authors2)
    t_batch2 = time.perf_counter()
    return {
        "device": nul.device,
        "embedding_model": nul.embedding_model,
        "student_inputs": nul.student_inputs,
        "init_ms": round((t_init - t0) * 1000, 1),
        "load_incl_warmup_ms": round((t_load - t0) * 1000, 1),
        "batch50_first_ms": round((t_batch - t_load) * 1000, 1),
        "batch50_second_ms": round((t_batch2 - t2) * 1000, 1),
        "batch50_cached_ms": round((t_cached - t_batch) * 1000, 2),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m gvon.classifier", description=__doc__.splitlines()[0])
    ap.add_argument("texts", nargs="*", help="texts to score")
    ap.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    ap.add_argument("--blocklist", default=str(DEFAULT_BLOCKLIST))
    ap.add_argument("--threshold", type=float, default=None,
                    help="override the threshold (default: GVON_THRESHOLD, else config.json, else 0.5)")
    ap.add_argument("--device", default=None, help="force torch device (sets GVON_DEVICE)")
    ap.add_argument("--bench", action="store_true", help="time model load and a 50-text batch")
    args = ap.parse_args(argv)
    if args.device:
        os.environ["GVON_DEVICE"] = args.device
    if args.bench:
        print(json.dumps(run_bench(args.model_dir, args.blocklist), indent=2))
        return 0
    if not args.texts:
        ap.error("give one or more texts, or --bench")
    nul = Nullifier(args.model_dir, args.blocklist, args.threshold)
    print(f"# threshold={nul.threshold} (source={nul.threshold_source}) text_model_enabled={nul.text_model_enabled}",
          file=sys.stderr)
    for text, p in zip(args.texts, nul.score(args.texts)):
        flag = "NULLIFY" if nul.text_model_enabled and p >= nul.threshold else "keep"
        if is_low_info(text):
            flag += " (abstain: low-info text)"
        print(f"{p:.4f}\t{flag}\t{text}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
