#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "$0")/.." && pwd)"

python "$DIR/scripts/preflight.py"

SESSION="rover-dev"

if command -v tmux >/dev/null 2>&1; then
  tmux new-session -d -s "$SESSION"
  tmux send-keys -t "$SESSION" "cd $DIR/services/asr && uvicorn services.asr.server:app --host 0.0.0.0 --port 4001 --reload" C-m
  tmux split-window -h -t "$SESSION"
  tmux send-keys -t "$SESSION" "cd $DIR/services/mt && uvicorn services.mt.server:app --host 0.0.0.0 --port 4002 --reload" C-m
  tmux split-window -v -t "$SESSION:0.0"
  tmux send-keys -t "$SESSION" "cd $DIR/backend && uvicorn backend.app:app --host 0.0.0.0 --port 4000 --reload" C-m
  tmux split-window -v -t "$SESSION:0.1"
  tmux send-keys -t "$SESSION" "cd $DIR/frontend && npm run dev" C-m
  tmux select-layout -t "$SESSION" tiled
  (xdg-open http://localhost:3000 >/dev/null 2>&1 || true) &
  tmux attach -t "$SESSION"
else
  echo "tmux not found; starting services sequentially" >&2
  (cd "$DIR/services/asr" && uvicorn services.asr.server:app --host 0.0.0.0 --port 4001 --reload &)
  (cd "$DIR/services/mt" && uvicorn services.mt.server:app --host 0.0.0.0 --port 4002 --reload &)
  (cd "$DIR/backend" && uvicorn backend.app:app --host 0.0.0.0 --port 4000 --reload &)
  (cd "$DIR/frontend" && npm run dev &)
  (xdg-open http://localhost:3000 >/dev/null 2>&1 || true) &
  wait
fi
