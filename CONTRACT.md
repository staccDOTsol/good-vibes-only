# GVON data contracts

Every stage reads/writes JSONL under `data/` (gitignored). These shapes are the interface between
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
`author_id, username, verdict ("block"|"watch"|"allow"), nullify_score (0..1), n_tweets, summary (str)`
## data/blocklist.json  (written by label)
`{"generated_at": ISO8601, "handle": GVON_HANDLE, "accounts": [{"id","username","nullify_score","summary"}]}`
Only verdict=="block" accounts.

## models/latest/  (written by train)
`config.json` = `{"embedding_model": str, "threshold": float, "trained_at": ISO8601, "n_train": int, "metrics": {...}}`
`head.joblib` = sklearn classifier over sentence embeddings (predict_proba column 1 == P(nullify)).

## gvon/classifier.py public API (used by nullifier + tests)
```python
class Nullifier:
    def __init__(self, model_dir: str | Path = "models/latest", blocklist_path: str | Path = "data/blocklist.json", threshold: float | None = None): ...
    def score(self, texts: list[str]) -> list[float]: ...            # P(nullify) per text, batched, cached by text hash
    def should_nullify(self, text: str, *, author_id: str | None = None, username: str | None = None) -> bool: ...
    # blocklist hit (by id or username, case-insensitive) => True regardless of text; else score >= threshold
```
Device selection: torch MPS if available, else CPU. Must load in < ~5s and score a 50-text batch in < ~200ms on an M-series Mac.
