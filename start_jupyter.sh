#!/usr/bin/env bash
# Standard Jupyter Lab launcher for the whole numerai folder (datasets +
# every research subfolder). Idempotent: does nothing if a server is already
# listening on the port. See JUPYTER.md for details.
set -euo pipefail

PORT=8888
VENV=~/ml-venv
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG="$PROJECT_ROOT/jupyter.log"

if ss -tln 2>/dev/null | grep -q "127.0.0.1:${PORT} "; then
  echo "Jupyter Lab already running on port ${PORT}."
  exit 0
fi

nohup "$VENV/bin/jupyter" lab \
  --no-browser \
  --ServerApp.token='' \
  --ServerApp.password='' \
  --ServerApp.allow_origin='*' \
  --ServerApp.root_dir="$PROJECT_ROOT" \
  --port "$PORT" \
  --ip 127.0.0.1 \
  > "$LOG" 2>&1 &
disown

sleep 2
echo "Started Jupyter Lab on http://127.0.0.1:${PORT} (root: $PROJECT_ROOT, log: $LOG)"
