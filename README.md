# Good Vibes Only Nullifier (GVON)

GVON is a Good Vibes Only Nullifier: a local, low-stack network classifier and nullifier for X, built so
the vile stuff never reaches your eyes. It reads a week of the replies, mentions and quotes aimed at your
account, has a frontier model judge every engaging account and post, distils those judgments into a
small model that runs on your own CPU or Apple Metal, and then strips matching posts out of x.com
responses before your browser renders them. No more block-whack-a-mole.

## Architecture

```
  X API v2 recent search          frontier "teacher"             local "student"
  (last 7 days, Basic tier)       (claude -p / Anthropic SDK)    (CPU or Apple Metal / MPS)
 +-----------------------+      +------------------------+     +-----------------------------+
 | gvon/ingest.py        | ---> | gvon/label.py          | --> | gvon/train.py               |
 | replies, mentions,    |      | per-tweet label:       |     | MiniLM sentence embeddings  |
 | quotes, own posts,    |      |  nullify|neutral|good  |     |  + logistic-regression head |
 | users, context        |      | per-author verdict:    |     | -> models/latest/           |
 | -> data/raw/*.jsonl   |      |  block|watch|allow     |     | gvon/classifier.py          |
 +-----------------------+      | -> data/labels/*.jsonl |     |  Nullifier.score()          |
                                | -> data/blocklist.json |     |  Nullifier.should_nullify() |
                                +------------------------+     +--------------+--------------+
                                                                              |
                         +----------------------------------------------------+
                         v                                                    v
          +-------------------------------------+          +-------------------------------+
          | gvon/nullifier.py (mitmproxy addon) |          | scripts/gvon-userscript.py    |
          | browser -> 127.0.0.1:8080 -> x.com  |          | -> dist/gvon.user.js          |
          | decrypts timeline JSON, prunes      |          | hides blocklisted authors     |
          | nullified entries (gvon/prune.py),  |          | after render (no CA, no       |
          | re-encodes; cursors untouched       |          | classifier; fallback only)    |
          +-------------------------------------+          +-------------------------------+
```

The interface between stages is defined in [CONTRACT.md](CONTRACT.md). Proxy setup, CA trust, browser
profiles and LAN/phone options are in [docs/NULLIFIER.md](docs/NULLIFIER.md).

## Quickstart

Prerequisites:

- [`uv`](https://docs.astral.sh/uv/) (used by `make setup` to create the Python 3.12 venv).
- An X API bearer token, Basic tier or higher (`X_BEARER_TOKEN`), and your handle (`GVON_HANDLE`).
- A teacher model. The default `GVON_TEACHER=claude-cli` needs the Claude Code CLI (`claude`) on your
  `PATH` and logged in. The alternative is `GVON_TEACHER=sdk` with `ANTHROPIC_API_KEY` set. `make label`
  stops with a clear error if neither is available.

```sh
cp .env.example .env          # fill in X_BEARER_TOKEN and GVON_HANDLE (your handle, no @)
make setup                    # uv venv (Python 3.12) + pip install -e ".[dev]"
make all                      # X ingest -> Telegram ingest (if configured) -> label -> train
                              #   (same as scripts/run-all.sh)
# trust the mitmproxy CA once: see docs/NULLIFIER.md section 2
make proxy                    # mitmproxy nullifier on port 8080
```

Then launch a browser profile that uses the proxy, so only that profile is filtered:

```sh
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
  --user-data-dir="$HOME/.gvon-chrome" --proxy-server="127.0.0.1:8080" --proxy-bypass-list="<-loopback>"
```

Log in to x.com in that window. Pruned posts are logged in the proxy terminal as `gvon: pruned ...`.

Telegram can feed the same teacher and student. Put `TG_API_ID` / `TG_API_HASH` (from my.telegram.org)
in `.env`, run `make telegram-login` once yourself (it asks for your phone number and the login code),
then `make telegram-ingest` pulls a week of DMs, replies and mentions into `data/raw/telegram.jsonl`.
`make label` labels every source that has data (`--source x|telegram|all`), and `make train` trains on
every labels file. The blocklist then holds X and Telegram accounts, each tagged with its `platform`.
`make all` (and `scripts/run-all.sh`) runs the Telegram ingest step only when `TG_API_ID`, `TG_API_HASH`
and a session (`data/telegram.session` or `TG_SESSION`) are all present. Otherwise it prints which one is
missing, skips Telegram and carries on with X only. It also skips the Telegram pull while the sink
(`make telegram`) is running, because one Telegram session cannot serve two processes: stop the sink, run
`make telegram-ingest`, then restart it. `make telegram` runs the Telegram sink ([docs/TELEGRAM.md](docs/TELEGRAM.md)).

Other targets:

| target | what it does |
| --- | --- |
| `make ingest` / `make label` / `make train` | run one stage |
| `make telegram-login` | one-time interactive Telegram login (you run it; phone + code) -> `data/telegram.session` |
| `make telegram-ingest` | pull a week of Telegram DMs / replies / mentions -> `data/raw/telegram*.jsonl` |
| `make telegram` | the Telegram sink: deletes, mutes or archives nullified incoming messages |
| `make telegram-vault N=20` | print the last N messages the Telegram sink nullified, to audit false positives |
| `make test` | full pytest suite (synthetic data only) |
| `make proxy-record` | proxy with `GVON_RECORD=1`: saves matched response bodies to `data/recordings/` |
| `make userscript` | writes `dist/gvon.user.js`, a blocklist-only fallback for Tampermonkey/Violentmonkey |
| `make clean-data` | deletes `data/raw`, `data/labels`, `data/recordings`, `data/blocklist.json`, `models/latest`, `dist` |

`make proxy` passes `--set block_global=false`, and mitmproxy listens on all interfaces by default.
On an untrusted network, firewall port 8080 or add `--listen-host 127.0.0.1` / `--proxyauth`, or anyone
who can reach the port can use your proxy.

Stages are resumable:

- **Ingest** saves raw pages and its pagination state, so a rerun continues where an API error (for
  example a 402 after credits ran out) stopped it. Recent search only accepts a start time within the
  last 7 days. When a saved start time has aged out of that window, ingest moves the window forward and
  continues each query from the oldest tweet it already has, so it keeps the saved pages and does not
  pay for them again. A completed run younger than 6 hours is reused rather than re-pulled.
  `make ingest INGEST_ARGS=--refresh` forces a new pull, and `INGEST_ARGS=--fresh` also discards
  unfinished state.
- **Label** caches per author, so a rerun only pays for authors whose tweets changed or failed. If at
  most 10% of the tweets fail (`--max-failure-frac`), it warns and exits 0 so `train` still runs.
  `make label LABEL_ARGS=--rebuild-only` rebuilds its outputs from the cache with no model calls.
- Ingest exits non-zero on an API error. That stops `make all` and `run-all.sh` before a partial week is
  labeled. Re-run once the problem is fixed. Because a recent completed ingest is reused, re-running
  `make all` after a later stage failed costs no X API reads.

## Apply it everywhere

The student model is platform-agnostic. `gvon.classifier.Nullifier` takes a post's text and, for the v2
student, an optional author dict (`bio`, `followers`, `following`, `tweet_count`, `created_at`, `verified`,
any of them missing). It has no idea which app the text came from. Training pools every labelled source
(`data/labels/tweets.jsonl` from X, `data/labels/telegram_tweets.jsonl` from Telegram), so one
`models/latest/` and one blocklist serve every sink. Each blocklist account carries a `platform`, and a
sink can scope identity matches to its own platform by passing `platform=` to `should_nullify`.

| sink | how it runs | what it can do | what it cannot do |
| --- | --- | --- | --- |
| x.com proxy (`gvon/nullifier.py`, `make proxy`) | mitmproxy addon between your browser and x.com | removes nullified posts from timeline, search, profile and conversation JSON before the browser renders them; uses the blocklist, the watchlist and the student (text plus author context) | native X apps (certificate pinning), DMs, Spaces and WebSocket traffic; needs a trusted local CA; scores text only on posts that engage you unless `GVON_TEXT_SCOPE=all`; passes no `platform`, so a blocked Telegram username also hides an X account with the same username |
| x.com userscript (`dist/gvon.user.js`, `make userscript`) | Tampermonkey / Violentmonkey in any browser | hides posts by blocklisted usernames after x.com renders them; no CA, no proxy, no Python at runtime | runs no classifier, so no per-post text scoring; posts are still downloaded and can flash before they are hidden; matches usernames only; must be regenerated after the blocklist changes; bakes in every blocklist username, Telegram ones included |
| Telegram sink (`gvon/telegram_nullifier.py`, `make telegram`) | a second Telegram client logged in as you (Telethon) | scores every incoming message with the same Nullifier (`platform="telegram"`); deletes it for you in DMs and basic groups (no read receipt is sent), mutes or archives the chat, or deletes it for everyone in groups (never broadcast channels) where you are an admin and enable it; keeps every nullified message in a local vault, attachments downloaded before any delete (a message whose attachment cannot be saved is only logged); never acts on Telegram's service account 777000, support accounts or `GVON_TG_ALLOW` senders | stop delivery: your other devices receive the message, and may show a push notification, before the sink acts; delete for you in supergroups or channels (it mutes instead); act while it is not running (`--backfill N` catches up on unread messages); score media without a caption or re-score edits |

**Adding a sink.** A new platform needs two pieces. First, an ingest module that writes the platform's
recent engagement in the same JSONL shape as `data/raw/tweets.jsonl`, plus a users file and a context file
(see [CONTRACT.md](CONTRACT.md); `gvon/telegram_ingest.py` is the worked example). Give rows a
`platform` key, register the source in `gvon/label.py` (a `SourceSpec` entry naming its raw, users and
context files, plus a rubric note for the teacher), and name its labels `data/labels/<prefix>_tweets.jsonl`
with raw rows in `data/raw/<prefix>.jsonl`, which `train` discovers on its own. Second, a sink that sees each item before
you do and calls `Nullifier.should_nullify(text, author_id=..., username=..., platform=..., author=...)`, then
hides, deletes or mutes whatever returns True. Nothing in the classifier needs to change.

## Honest limitations

- **HTTPS means per-post filtering needs a trusted local CA.** The proxy has to decrypt x.com's TLS to
  read and edit the timeline JSON. DNS or hosts-file blocking can only block all of X or none of it. You
  must trust a CA that only your machine holds. Its private key, `~/.mitmproxy/mitmproxy-ca.pem`, must be
  treated like a password.
- **Mobile apps pin their certificates.** The native X apps reject the mitmproxy CA, so they cannot be
  filtered. Only x.com in a browser is covered, on desktop or in a phone browser.
- **The X API Basic tier (or higher) is needed for search.** Ingest uses v2 recent search, which the
  free tier does not include. Usage is metered: when credits run out, the API returns HTTP 402 and
  ingest stops with a partial dataset.
- **The search window is 7 days.** Recent search only reaches back one week, so the training set is
  whatever happened to you that week.
- **The teacher still sees more than the student.** The teacher judged each post with the author's bio,
  account age, follower counts, the parent tweet and everything else that author posted. The default
  student (v2) sees the post text plus the cheap part of that context: the bio, follower, following and
  post counts, account age and the verified flag, from `data/raw/users.jsonl` at training time and from the
  timeline JSON at filtering time. It never sees the parent post or the author's other posts.
  `python -m gvon.train --student-inputs text` trains the text-only v1 student instead. `student_inputs` in
  `models/latest/config.json` says which one is installed. Telegram has no public follower counts or
  account age. The Telegram sink fetches a sender's bio once per sender (one `GetFullUser` call, cached for
  a day). Bot, shill and copy-paste spam is still left to the blocklist: training drops nullify labels whose
  reason is identity-based (`spam_bot`, `shill`, `generic_filler`, `engagement_farming`) and learns only
  from content reasons such as hostility, insults, FUD, rug or scam insinuations, entitled demands and
  sneering.
- **The classifier is only as good as your labeled week.** The student learns from a few hundred
  teacher-labelled posts. It inherits the teacher's judgment calls and that week's topics, and it will
  miss new kinds of abuse until you ingest, label and train again. The metrics in
  `models/latest/config.json` are out-of-fold and grouped by author. They estimate performance on
  accounts the model has never seen, and their bootstrap CIs are wide.
- **Text scoring can be switched off by training.** Training picks the threshold that reaches
  out-of-fold precision `GVON_MIN_PRECISION` (default 0.7). If no threshold reaches it, training writes
  `text_model_enabled: false` and prints a warning, and the proxy then filters by the blocklist only.
  Setting `GVON_THRESHOLD` (in the environment or `.env`) is an explicit override: it switches text
  scoring back on at that threshold, for the proxy and the classifier CLI alike.
- **Text is only scored on engagement with you.** By default the proxy scores a post's text only when it
  replies to you, mentions you or quotes you, because that is all the model was trained on. Every other
  post (home timeline, search, profiles) is checked against the blocklist only. `GVON_TEXT_SCOPE=all`
  scores everything. False positives there have not been measured.
- **Authors get three tiers, and the teacher decides them.** By default (`GVON_VERDICT_POLICY=teacher`)
  the frontier teacher's per-author verdict is operative, so one scam accusation can put an account on
  `block` (data/blocklist.json); `watch` (data/watchlist.json) is for mixed accounts and single mild hits.
  `GVON_VERDICT_POLICY=derived` (or `make label LABEL_ARGS="--rebuild-only --verdict-policy derived"`)
  switches to a stricter code rule: at least two nullify posts that make up most of what the author
  posted, or one severe hit. Both verdicts are stored in authors.jsonl, so switching needs no relabel.
  Watch-listed authors get a threshold lowered by `GVON_WATCH_DELTA` (default 0.15). Any author whose
  running mean over 3 or more distinct posts reaches the threshold is hidden.
- X changes its web client often. The pruner is structure-agnostic and passes a response through
  untouched if anything goes wrong, so expect some misses rather than broken pages. DMs, Spaces,
  WebSocket traffic and the native apps are not filtered.
- Cold start takes several seconds, mostly Python importing torch and transformers. After that, a
  50-post batch scores in tens of milliseconds on an M-series Mac.

## Privacy

- **Everything stays local.** Raw tweets, labels, the blocklist, trained models and proxy recordings live
  under `data/` and `models/`, and both are gitignored. The classifier runs on your CPU or GPU. The proxy
  makes no network requests of its own and sends no telemetry. The labeling step sends the week's
  engagement text to the teacher model you configure (`GVON_TEACHER`), and nothing else leaves the machine.
- **The blocklist is YOUR opinion of YOUR timeline.** `data/blocklist.json` and the generated userscript
  record a model's judgment of specific people, made for your own filtering. Publish or share them at
  your own risk.
- The fixtures in `fixtures/` and all test data are synthetic, not real X data. Never copy a proxy
  recording (`data/recordings/`) into `fixtures/`. Hand-write a synthetic fixture that mimics the
  recording's structure, with invented text, ids and handles.

## Rotate your token

Your X bearer token belongs only in `.env`, which is gitignored and never printed or logged by GVON. If
it was ever pasted into a chat, a terminal recording, an issue, a commit, or shared with anyone, revoke
and regenerate it in the X developer portal (your app -> Keys and tokens -> Bearer token ->
Regenerate), then update `.env`. A leaked bearer token can spend your API credits.
