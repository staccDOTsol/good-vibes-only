# GVON data contracts

Every stage reads/writes JSONL under `data/` (gitignored). Default paths are anchored at the repo root
(`gvon.env.ROOT`, where `.env` lives), never the current working directory. These shapes are the interface between
`gvon/ingest.py` (X) and `gvon/telegram_ingest.py` (Telegram) -> `gvon/label.py` -> `gvon/train.py` ->
`gvon/classifier.py` -> `gvon/nullifier.py`.

## data/raw/tweets.jsonl  (written by ingest)
One object per tweet that engaged with GVON_HANDLE in the last 7 days (replies, mentions, quotes),
plus the handle's own tweets (kind="own") so labelers have context. Keys:
```
id, text, author_id, author_username, author_name, created_at (ISO8601), conversation_id,
in_reply_to_user_id (or null), lang, kind ("reply"|"mention"|"quote"|"own"),
referenced (list of {type: "replied_to"|"quoted"|"retweeted", id}),
public_metrics ({like_count, reply_count, retweet_count, quote_count, impression_count?})
```
## data/raw/users.jsonl  (written by ingest)
`id, username, name, description, created_at, public_metrics ({followers_count, following_count, tweet_count}), verified (bool)`
## data/raw/context.jsonl  (written by ingest)
Referenced tweets fetched via expansions (same shape as tweets.jsonl, kind="context"). Used so a label
prompt can show "X replied to <GVON_HANDLE's tweet>: <text>".

## data/raw/telegram.jsonl  (written by telegram_ingest)
Telegram messages from the last `--since-days` (default 7), same keys as tweets.jsonl where meaningful:
```
id (f"{chat_id}:{msg_id}", chat_id = Telethon marked peer id), text (raw message text, "" for media-only),
author_id (sender peer id as str), author_username (or null), author_name, created_at (ISO8601 UTC),
conversation_id (chat id as str), in_reply_to_user_id (sender of the replied-to message, or null),
lang (null), kind ("dm"|"group_reply"|"group_mention"|"group"|"own"),
referenced ([{type: "replied_to", id: "<chat_id>:<msg_id>"}] or []), public_metrics ({}),
platform ("telegram"), chat_title, chat_type ("private"|"bot"|"group"|"supergroup"), media (e.g. "photo", or null)
```
Kinds: dm = incoming message in a private chat; group_reply = reply to your message (or to a post of a
channel you own, in its discussion group); group_mention = message that mentions you; group = any other
message in a group where you posted in the window; own = your messages (context only, never labeled).
Groups where you did not post contribute only mentions/replies to you. Never collected: Saved Messages,
Telegram's service account 777000 (login codes), broadcast channels you do not own. At most
`--max-per-chat` (300) rows per chat (dm/group_reply first, then group_mention, own, group); drops are
counted in data/raw/_telegram_summary.json. A chat Telegram refuses to read (any RPCError other than a flood
wait, e.g. ChannelPrivateError) is skipped, its state not advanced, and counted in the summary's
`chats_failed` ({error name: count}); ChannelForbidden/ChatForbidden dialogs count as `chats_skipped.forbidden`.
## data/raw/telegram_users.jsonl  (written by telegram_ingest)
`id, username (or null), name, description (null: bios are not fetched), platform ("telegram"),
bot, scam, fake, verified, premium (bools), is_self (true only for the logged-in account)`
## data/raw/telegram_context.jsonl  (written by telegram_ingest)
Replied-to messages that are not themselves rows of telegram.jsonl (same shape, kind="context").
## data/raw/_telegram_state.json  (written by telegram_ingest)
`{"version": 1, "account_id": int, "chats": {"<chat_id>": {"last_id": int, "type": str, "updated_at": ISO8601}}}`:
the highest message id examined per chat; the next pull asks only for newer messages (min_id) and merges.
A state written for another account is discarded. `pull --fresh` ignores it.
## data/telegram.session  (written by `telegram_ingest login`)
Telethon SQLite session = a full login to the Telegram account; under gitignored data/, chmod 600.
`TG_SESSION` (a Telethon StringSession) is accepted instead by `pull` and the sink (never by `login`).
TG_API_ID / TG_API_HASH come from .env. Every TelegramClient is built by `gvon.telegram_client.make_client()`
(session: TG_SESSION > explicit path > GVON_TG_SESSION > data/telegram.session). Session paths follow Telethon:
".session" is appended when missing (`--session mytg` -> mytg.session).
## data/telegram.session.lock  (held by the sink, `pull` and `login`)
Exclusive fcntl.flock for the life of a client (data/telegram.tg_session.lock when TG_SESSION is used); content
`"<holder> pid=<pid>"`, no secrets. A second process raises `gvon.telegram_client.SessionBusyError` (a
TelegramConfigError: exit 2). `python -m gvon.telegram_client ready` exits 0 when creds and a session are present
and the session is not locked, 3 otherwise, including "in use by sink" (`make all` / run-all.sh skip Telegram on 3).
`gvon.telegram_client.SERVICE_USER_IDS` ({777000}) is never collected by the ingest and never acted on by the sink.

## data/labels/tweets.jsonl  (written by label, source "x")
`id, author_id, author_username, label ("nullify"|"neutral"|"good"), nullify_score (0..1 float), reasons (list[str])`
## data/labels/authors.jsonl  (written by label, source "x")
`author_id, username, verdict ("block"|"watch"|"allow"), nullify_score (0..1), n_tweets, summary (str),
teacher_verdict, teacher_nullify_score, derived_verdict, derived_nullify_score,
verdict_policy ("teacher"|"derived"), platform ("x"|"telegram")`
Both verdicts are always stored. `teacher_verdict` / `teacher_nullify_score` are the frontier teacher's
per-author judgement. `derived_verdict` / `derived_nullify_score` come from code (gvon.label.derive_verdict):
block = >= 2 nullify tweets that are >= 60% of the author's tweets, or one severe hit (first reason
slur/threat/hostility/insult with score >= 0.9); watch = any other nullify tweet; allow = none;
score = max(mean, 0.5 * max) of the tweet scores.
`verdict` / `nullify_score` are the OPERATIVE pair, picked by the verdict policy (`--verdict-policy` >
`GVON_VERDICT_POLICY` > "teacher"): "teacher" (default) = the teacher's pair, so one scam accusation can
block; "derived" = the code rule. `--rebuild-only` re-applies a policy from the cache with no model calls.
## data/labels/telegram_tweets.jsonl, data/labels/telegram_authors.jsonl  (written by label, source "telegram")
Same shapes as tweets.jsonl / authors.jsonl (platform "telegram"). `label --source x|telegram|all`
(default all: every source with raw rows). Per-source cache/summary/failures: data/labels/_cache.jsonl,
_summary.json, _failures.jsonl (X) and _telegram_cache.jsonl, _telegram_summary.json, _telegram_failures.jsonl.
## data/blocklist.json  (written by label)
`{"generated_at": ISO8601, "handle": GVON_HANDLE, "accounts": [{"id","username","nullify_score","summary","platform"}]}`
Only verdict=="block" accounts, from EVERY source's authors file (a `--source x` run keeps the Telegram
accounts and vice versa). `platform` is "x" or "telegram"; an account without it is an X account.
Telegram accounts without a public username have `username` "".
## data/watchlist.json  (written by label, next to blocklist.json)
Same shape as blocklist.json, holding only verdict=="watch" accounts. A missing file means nobody is watched.

## models/latest/  (written by train)
`config.json` = `{"embedding_model": str, "threshold": float, "trained_at": ISO8601, "n_train": int, "metrics": {...},
"threshold_source": "cli"|"env"|"oof_precision_target"|"none_met_precision_target", "text_model_enabled": bool,
"oof_threshold": float|null, "oof_precision_target_met": bool, "min_precision": float,
"student_inputs": ["tweet_text"] (v1) | ["tweet_text", "author_bio", "author_stats"] (v2),
"student_inputs_requested": "auto"|"text"|"author", "feature_dim": int,
"author_features": null (v1) | {"bio", "scalars": [names], "missing_indicators": [names], "scaler",
  "account_age_reference", "users_files": {source: {"path", "exists"}},
  "coverage_by_source": {source: {"rows", "with_author", "with_bio"}}},
"label_counts": {...}, "label_counts_by_source": {"x": {...}, "telegram": {...}},
"label_sources": [{"source", "labels_path", "raw_path", "n_label_rows", "n_train"}], ...}`
Inputs: every `data/labels/<prefix>tweets.jsonl` that exists (or the `--labels` files), each joined with its raw
file by prefix (tweets.jsonl <- data/raw/tweets.jsonl, telegram_tweets.jsonl <- data/raw/telegram.jsonl, or one
`--raw` per `--labels`). Non-X author ids are namespaced ("telegram:<id>") before author grouping.
`metrics` are out-of-fold predictions under author-grouped CV (`metrics.split == "grouped_by_author"`).
`text_model_enabled == false` (threshold 1.01) means no threshold reached the precision target, so text scoring is off.
`head.joblib` = sklearn classifier (LogisticRegression + Platt calibrator) over the feature row below
(predict_proba column 1 == P(nullify)).
Feature row: v1 = MiniLM embedding (384, L2-normalised) of `gvon.classifier.content_text(text)`.
v2 = that ++ MiniLM embedding of the author's bio (whitespace-collapsed; zeros when empty or unknown)
++ `AUTHOR_SCALAR_NAMES` (log1p_followers, log1p_following, log1p_tweet_count, log1p_account_age_days,
log1p_followers_per_following = log1p(followers / (following + 1)), verified, has_bio) standardised by
`author_scaler.joblib` (sklearn StandardScaler; an unknown value is NaN before scaling and 0 after)
++ `AUTHOR_MISSING_NAMES` 0/1 indicators (followers, following, tweet_count, account_age, verified).
No NaN ever reaches the head. Account age = post created_at - account created_at at train time,
scoring time - account created_at at inference.
Author context at train time (v2): X rows from `data/raw/users.jsonl` (description, public_metrics,
created_at, verified), Telegram rows from `data/raw/telegram_users.jsonl` (description is null, counts and
creation date do not exist, so only verified is known). `--student-inputs auto` (default) trains v2 when any
source has its users file (users.jsonl / telegram_users.jsonl next to its raw file), else v1;
`text` forces v1, `author` forces v2. A v1 train into a directory removes a stale `author_scaler.joblib`.

## gvon/classifier.py public API (used by nullifier + tests)
```python
class Nullifier:
    def __init__(self, model_dir: str | Path = ROOT/"models/latest", blocklist_path: str | Path = ROOT/"data/blocklist.json",
                 threshold: float | None = None, watchlist_path: str | Path | None = None): ...  # None => <blocklist dir>/watchlist.json
    threshold: float; threshold_source: str; text_model_enabled: bool
    student_inputs: list[str]; uses_author: bool; uses_author_bio: bool  # from config.json
    def score(self, texts: list[str], authors: list[dict | None] | None = None) -> list[float]: ...
        # P(nullify), batched, cached by cleaned-text hash (+ author signature for v2); 0.0 (abstain) when
        # < 3 content chars. authors aligned with texts; ignored by v1; None / missing fields = unknown.
    def is_blocked(self, *, author_id=None, username=None, platform=None) -> bool: ...
    def is_watched(self, *, author_id=None, username=None, platform=None) -> bool: ...
    def should_nullify(self, text: str, *, author_id: str | None = None, username: str | None = None,
                       platform: str | None = None, author: dict | None = None) -> bool: ...
    # author (optional, additive) = {bio, followers, following, tweet_count, created_at, verified}; any key
    # may be missing or None. created_at: ISO 8601 or X legacy "Wed Oct 10 20:19:24 +0000 2018".
    # platform=None matches block/watch accounts of ANY platform (what the X proxy and older callers do);
    # "x" / "telegram" match only that platform's accounts (entries without "platform" count as "x").
    # blocklist hit (by id or username, case-insensitive) => True regardless of text;
    # else False if text_model_enabled is False or the text is low-info;
    # else score >= threshold (threshold - GVON_WATCH_DELTA for watch-listed authors),
    # or the author's running mean over >= 3 distinct scored texts >= threshold.
```
`gvon.classifier.load_blocklist(path, platform=None) -> (ids, usernames)` follows the same platform rule.
Threshold resolution (`gvon.classifier.resolve_threshold`, used by the proxy and the CLI alike): explicit
argument > GVON_THRESHOLD > config.json > 0.5. An explicit argument or GVON_THRESHOLD also forces
text_model_enabled on.
Device selection: torch MPS if available, else CPU. Must load in < ~5s and score a 50-text batch in < ~200ms on an M-series Mac.

## gvon/prune.py public API (used by nullifier)
```python
def prune(payload, decider, *, owner_ids=(), owner_usernames=(), cache=None,
          text_scope: str = "engagements") -> tuple[new_payload, list[{"entryId","username","reason"}]]
```
`new_payload is payload` exactly when nothing changed (callers re-encode only when it is a new object).
When `decider.should_nullify` declares an `author` parameter (and `score` an `authors` parameter), each judged
tweet's author context is passed: GraphQL user result -> {bio: legacy.description (or profile_bio.description),
followers: legacy.followers_count, following: legacy.friends_count, tweet_count: legacy.statuses_count,
created_at: legacy.created_at (or core.created_at), verified: is_blue_verified or legacy.verified
(or verification.verified), None if none present}; REST globalObjects.users[id] -> the same from description,
followers_count, friends_count, statuses_count, created_at, verified / is_blue_verified. Deciders without
those parameters are called exactly as before.
text_scope="engagements": tweet text goes to `decider.should_nullify` only for tweets that reply to,
mention or quote the owner; every tweet is still checked by identity (`is_blocked`). "all": every tweet.
