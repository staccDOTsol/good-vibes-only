#!/usr/bin/env bash
# Run the full GVON pipeline in order: X ingest -> Telegram ingest -> label -> train.
# The Telegram step runs only when TG_API_ID/TG_API_HASH are set, a session exists (data/telegram.session from
# `make telegram-login`, or TG_SESSION) and the sink is not holding that session; otherwise it is skipped
# with the reason.
# Stops at the first failing stage (e.g. ingest exits 1 on an X API error such as 402 credits-depleted;
# whatever real pages were fetched are still written to data/raw/ and the next run resumes).
# Re-running after a later stage failed is cheap: ingest reuses a completed run younger than 6h instead of
# re-pulling (and re-billing) the week, and label only pays for authors missing from its cache.
# Extra flags: INGEST_ARGS / TG_INGEST_ARGS / LABEL_ARGS / TRAIN_ARGS environment variables (e.g. INGEST_ARGS=--refresh).
set -euo pipefail

cd "$(dirname "$0")/.."
PY=.venv/bin/python

if [[ ! -x "$PY" ]]; then
  echo "missing $PY; run 'make setup' first" >&2
  exit 2
fi

echo "== ingest =="
"$PY" -m gvon.ingest ${INGEST_ARGS:-}
echo "== telegram-ingest =="
if "$PY" -m gvon.telegram_client ready; then
  "$PY" -m gvon.telegram_ingest pull ${TG_INGEST_ARGS:-}
else
  echo "== telegram-ingest skipped: see the reason above (X-only run; set TG_API_ID/TG_API_HASH in .env and run 'make telegram-login' to include Telegram) =="
fi
echo "== label =="
"$PY" -m gvon.label ${LABEL_ARGS:-}
echo "== train =="
"$PY" -m gvon.train ${TRAIN_ARGS:-}
echo "== done: models/latest/ ready; start the proxy with 'make proxy' and/or the Telegram sink with 'make telegram' =="
