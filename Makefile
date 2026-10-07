# GVON pipeline. Every stage reads/writes local files under data/ and models/ (both gitignored).
# Order: ingest (X API) -> label (frontier teacher) -> train (local student) -> proxy / userscript.

PY      := .venv/bin/python
MITM    := .venv/bin/mitmdump
PORT    ?= 8080
# extra flags per stage, e.g. `make ingest INGEST_ARGS=--refresh` or `make label LABEL_ARGS="--limit-authors 20"`
INGEST_ARGS ?=
LABEL_ARGS  ?=
TRAIN_ARGS  ?=

.PHONY: setup ingest label train test proxy proxy-record userscript all clean-data

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

## all: ingest -> label -> train
all: ingest label train

## clean-data: delete ALL local data, labels, blocklist, recordings and trained models
clean-data:
	rm -rf data/raw data/labels data/recordings data/blocklist.json models/latest dist
