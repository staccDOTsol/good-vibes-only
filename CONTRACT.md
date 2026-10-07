# GVON data contracts

Every stage reads/writes JSONL under `data/` (gitignored). Default paths are anchored at the repo root
(`gvon.env.ROOT`, where `.env` lives), never the current working directory. These shapes are the interface between
`gvon/ingest.py` -> `gvon/label.py` -> `gvon/train.py` -> `gvon/classifier.py` -> `gvon/nullifier.py`.

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

## data/labels/tweets.jsonl  (written by label)
`id, author_id, author_username, label ("nullify"|"neutral"|"good"), nullify_score (0..1 float), reasons (list[str])`
## data/labels/authors.jsonl  (written by label)
`author_id, username, verdict ("block"|"watch"|"allow"), nullify_score (0..1), n_tweets, summary (str),
teacher_verdict, teacher_nullify_score`
`verdict` and `nullify_score` are derived in code from that author's tweet labels (gvon.label.derive_verdict):
block = >= 2 nullify tweets that are >= 60% of the author's tweets, or one severe hit (first reason
slur/threat/hostility/insult with score >= 0.9); watch = any other nullify tweet; allow = none.
nullify_score = max(mean, 0.5 * max) of the tweet scores. The teacher's own values are kept as
`teacher_verdict` / `teacher_nullify_score` (advisory).
## data/blocklist.json  (written by label)
`{"generated_at": ISO8601, "handle": GVON_HANDLE, "accounts": [{"id","username","nullify_score","summary"}]}`
Only verdict=="block" accounts.
## data/watchlist.json  (written by label, next to blocklist.json)
Same shape as blocklist.json, holding only verdict=="watch" accounts. A missing file means nobody is watched.

## models/latest/  (written by train)
`config.json` = `{"embedding_model": str, "threshold": float, "trained_at": ISO8601, "n_train": int, "metrics": {...},
"threshold_source": "cli"|"env"|"oof_precision_target"|"none_met_precision_target", "text_model_enabled": bool,
"oof_threshold": float|null, "oof_precision_target_met": bool, "min_precision": float, "student_inputs": ["tweet_text"], ...}`
`metrics` are out-of-fold predictions under author-grouped CV (`metrics.split == "grouped_by_author"`).
`text_model_enabled == false` (threshold 1.01) means no threshold reached the precision target, so text scoring is off.
`head.joblib` = sklearn classifier over sentence embeddings of `gvon.classifier.content_text(text)`
(predict_proba column 1 == P(nullify)).

## gvon/classifier.py public API (used by nullifier + tests)
```python
class Nullifier:
    def __init__(self, model_dir: str | Path = ROOT/"models/latest", blocklist_path: str | Path = ROOT/"data/blocklist.json",
                 threshold: float | None = None, watchlist_path: str | Path | None = None): ...  # None => <blocklist dir>/watchlist.json
    threshold: float; threshold_source: str; text_model_enabled: bool
    def score(self, texts: list[str]) -> list[float]: ...            # P(nullify) of content_text(text), batched, cached by
                                                                      # cleaned-text hash; 0.0 (abstain) when < 3 content chars
    def is_blocked(self, *, author_id=None, username=None) -> bool: ...
    def is_watched(self, *, author_id=None, username=None) -> bool: ...
    def should_nullify(self, text: str, *, author_id: str | None = None, username: str | None = None) -> bool: ...
    # blocklist hit (by id or username, case-insensitive) => True regardless of text;
    # else False if text_model_enabled is False or the text is low-info;
    # else score >= threshold (threshold - GVON_WATCH_DELTA for watch-listed authors),
    # or the author's running mean over >= 3 distinct scored texts >= threshold.
```
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
text_scope="engagements": tweet text goes to `decider.should_nullify` only for tweets that reply to,
mention or quote the owner; every tweet is still checked by identity (`is_blocked`). "all": every tweet.
