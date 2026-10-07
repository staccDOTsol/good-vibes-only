#!/usr/bin/env bash
# Run the full GVON pipeline in order: ingest -> label -> train.
# Stops at the first failing stage (e.g. ingest exits 1 on an X API error such as 402 credits-depleted;
# whatever real pages were fetched are still written to data/raw/ and the next run resumes).
# Re-running after a later stage failed is cheap: ingest reuses a completed run younger than 6h instead of
# re-pulling (and re-billing) the week, and label only pays for authors missing from its cache.
# Extra flags: INGEST_ARGS / LABEL_ARGS / TRAIN_ARGS environment variables (e.g. INGEST_ARGS=--refresh).
set -euo pipefail

cd "$(dirname "$0")/.."
PY=.venv/bin/python

if [[ ! -x "$PY" ]]; then
  echo "missing $PY; run 'make setup' first" >&2
  exit 2
fi

echo "== ingest =="
"$PY" -m gvon.ingest ${INGEST_ARGS:-}
echo "== label =="
"$PY" -m gvon.label ${LABEL_ARGS:-}
echo "== train =="
"$PY" -m gvon.train ${TRAIN_ARGS:-}
echo "== done: models/latest/ ready; start the proxy with 'make proxy' =="
