#!/usr/bin/env bash
# Start (or reattach to) the development tmux session.
#   bash scripts/dev-tmux.sh
set -euo pipefail

S=nsr                                     # session name
ROOT="${PWD}"
PORT="${JUPYTER_PORT:-8888}"
ACT="source ${ROOT}/.venv/bin/activate"

if tmux has-session -t "$S" 2>/dev/null; then
  exec tmux attach -t "$S"
fi

# --- window 0: Claude Code, full height -------------------------------------
tmux new-session -d -s "$S" -c "$ROOT" -n claude
tmux send-keys -t "$S:claude" "$ACT && claude" C-m

# --- window 1: shell (top) + test runner (bottom) ---------------------------
tmux new-window -t "$S" -c "$ROOT" -n shell
tmux send-keys -t "$S:shell" "$ACT && git status -sb" C-m
tmux split-window -v -p 40 -t "$S:shell" -c "$ROOT"
tmux send-keys -t "$S:shell" "$ACT && pytest" C-m
tmux select-pane -t "$S:shell" -U

# --- window 2: Jupyter Lab server -------------------------------------------
tmux new-window -t "$S" -c "$ROOT" -n jupyter
tmux send-keys -t "$S:jupyter" \
  "$ACT && jupyter lab --no-browser --port ${PORT}" C-m

# --- quality of life ---------------------------------------------------------
tmux set-option -t "$S" mouse on          # scroll + click panes with the mouse
tmux set-option -t "$S" history-limit 50000
tmux set-option -t "$S" -g status-left "#[bold] $S "

tmux select-window -t "$S:claude"
exec tmux attach -t "$S"
