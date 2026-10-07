# GVON pipeline. Every stage reads/writes local files under data/ and models/ (both gitignored).
# Order: ingest (X API) + telegram-ingest (Telethon) -> label (frontier teacher) -> train (local student)
# -> sinks: proxy / userscript (x.com), telegram (Telegram).

PY      := .venv/bin/python
MITM    := .venv/bin/mitmdump
PORT    ?= 8080
# extra flags per stage, e.g. `make ingest INGEST_ARGS=--refresh` or `make label LABEL_ARGS="--limit-authors 20"`
INGEST_ARGS ?=
LABEL_ARGS  ?=
TRAIN_ARGS  ?=
# e.g. `make telegram-ingest TG_INGEST_ARGS="--chats 50"`
TG_INGEST_ARGS ?=
# e.g. `make telegram TG_ARGS="--backfill 50 --dry-run"`
TG_ARGS     ?=

.PHONY: setup ingest label train test proxy proxy-record userscript all clean-data telegram telegram-vault telegram-login telegram-ingest telegram-ingest-if-configured

## setup: create the uv-managed Python 3.12 venv and install gvon + dev deps
setup:
	test -x .venv/bin/python || uv venv --python 3.12 .venv
	VIRTUAL_ENV=.venv uv pip install -e ".[dev]"

## ingest: pull the last 7 days of engagements via X API v2 recent search (needs X_BEARER_TOKEN in .env).
##   Resumes an unfinished run; reuses a completed run younger than 6h (INGEST_ARGS=--refresh re-pulls,
##   INGEST_ARGS=--fresh discards saved state and pages).
ingest:
	$(PY) -m gvon.ingest $(INGEST_ARGS)

## label: frontier teacher labels every engaging author + tweet, writes data/blocklist.json
label:
	$(PY) -m gvon.label $(LABEL_ARGS)

## train: distil labels into the local student (MiniLM embeddings + logistic head) -> models/latest/
train:
	$(PY) -m gvon.train $(TRAIN_ARGS)

## test: full test suite (synthetic data only)
test:
	$(PY) -m pytest -q

## proxy: run the mitmproxy nullifier on 127.0.0.1:$(PORT) (loopback only; override LISTEN_HOST for LAN use, then firewall it)
##   (listen_host is empty), and block_global=false also admits public-IP clients: anyone who can reach
##   this port can use the proxy. Firewall the port on untrusted networks (see README).
proxy:
	$(MITM) -s gvon/nullifier.py --listen-host 127.0.0.1 --listen-port $(PORT) --set block_global=false

## proxy-record: same, but save matched response bodies to data/recordings/ for fixtures
proxy-record:
	GVON_RECORD=1 $(MITM) -s gvon/nullifier.py --listen-host 127.0.0.1 --listen-port $(PORT) --set block_global=false

## userscript: write dist/gvon.user.js (blocklist-only browser fallback, no CA needed)
userscript:
	$(PY) scripts/gvon-userscript.py

## telegram-login: one-time interactive Telegram login (phone + code, 2FA password if set) -> data/telegram.session
telegram-login:
	$(PY) -m gvon.telegram_ingest login

## telegram-ingest: pull the last 7 days of Telegram DMs / group replies / mentions -> data/raw/telegram*.jsonl
##   (resumable via data/raw/_telegram_state.json; `make label` then labels them with --source all).
##   Refuses to run while the sink (`make telegram`) uses the same session: stop the sink first.
telegram-ingest:
	$(PY) -m gvon.telegram_ingest pull $(TG_INGEST_ARGS)

## telegram: Telegram sink, runs as you and deletes/mutes/archives nullified incoming messages (docs/TELEGRAM.md).
##   Needs TG_API_ID/TG_API_HASH in .env and a session from `make telegram-login`. Vault: data/telegram_nullified.jsonl
telegram:
	$(PY) -m gvon.telegram_nullifier $(TG_ARGS)

## telegram-vault: print the last N (default 20) nullified Telegram messages to audit false positives
N ?= 20
telegram-vault:
	$(PY) -m gvon.telegram_nullifier --vault $(N)

## telegram-ingest-if-configured: telegram-ingest when TG_API_ID/TG_API_HASH and a session (data/telegram.session
##   or TG_SESSION) are present and no other gvon process (e.g. the running sink) holds that session; otherwise
##   prints why and skips (exit 0). Used by `make all`.
telegram-ingest-if-configured:
	@if $(PY) -m gvon.telegram_client ready; then \
		$(PY) -m gvon.telegram_ingest pull $(TG_INGEST_ARGS); \
	else \
		echo "== telegram-ingest skipped: see the reason above (X-only run; set TG_API_ID/TG_API_HASH in .env and run 'make telegram-login' to include Telegram) =="; \
	fi

## all: X ingest -> Telegram ingest (skipped unless configured) -> label (every source) -> train
##   Serial even under `make -j`: each stage needs the previous one's output.
all:
	$(MAKE) ingest
	$(MAKE) telegram-ingest-if-configured
	$(MAKE) label
	$(MAKE) train

## clean-data: delete ALL local data, labels, blocklist, recordings and trained models
clean-data:
	rm -rf data/raw data/labels data/recordings data/blocklist.json models/latest dist
