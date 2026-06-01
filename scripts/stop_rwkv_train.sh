#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/stop_rwkv_train.sh <launcher-pid>

Example:
  bash scripts/stop_rwkv_train.sh "$(pgrep -fo 'python .*train\.py')"

This script sends signals to the target process group in three steps:
  1. SIGINT  - let torchrun / train.py tear down workers cleanly
  2. SIGTERM - if workers are still alive after the grace period
  3. SIGKILL - only as a last resort
EOF
}

if [[ $# -ne 1 ]]; then
  usage
  exit 1
fi

pid="$1"
if ! [[ "$pid" =~ ^[0-9]+$ ]]; then
  echo "error: PID must be a number: $pid" >&2
  exit 1
fi

if ! ps -p "$pid" >/dev/null 2>&1; then
  echo "error: PID $pid does not exist" >&2
  exit 1
fi

cmd="$(ps -o cmd= -p "$pid")"
if [[ "$cmd" != *"train.py"* && "$cmd" != *"torch.distributed.run"* ]]; then
  echo "error: PID $pid does not look like an RWKV train launcher" >&2
  echo "cmd: $cmd" >&2
  exit 1
fi

pgid="$(ps -o pgid= -p "$pid" | tr -d '[:space:]')"
if ! [[ "$pgid" =~ ^[0-9]+$ ]]; then
  echo "error: failed to resolve PGID for PID $pid" >&2
  exit 1
fi

wait_for_exit() {
  local target_pgid="$1"
  local timeout_sec="$2"
  local elapsed=0

  while [[ "$elapsed" -lt "$timeout_sec" ]]; do
    if ! pgrep -g "$target_pgid" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
    elapsed=$((elapsed + 1))
  done

  return 1
}

echo "Stopping RWKV training launcher PID=$pid PGID=$pgid"
echo "Command: $cmd"

echo "Step 1/3: SIGINT to process group $pgid"
kill -INT -- "-$pgid"
if wait_for_exit "$pgid" 30; then
  echo "Training exited cleanly after SIGINT"
  exit 0
fi

echo "Step 2/3: SIGTERM to process group $pgid"
kill -TERM -- "-$pgid"
if wait_for_exit "$pgid" 15; then
  echo "Training exited after SIGTERM"
  exit 0
fi

echo "Step 3/3: SIGKILL to process group $pgid"
kill -KILL -- "-$pgid" || true
if wait_for_exit "$pgid" 5; then
  echo "Training force-killed"
  exit 0
fi

echo "warning: some processes in PGID $pgid may still be alive" >&2
exit 1
