# Sourced by the local helper scripts. Reads LUMI settings from .env.local (never committed).
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$REPO/.env.local"
: "${LUMI_HOST:?}" "${LUMI_PROJECT:?}" "${LUMI_PROJECT_DIR:?}" "${LUMI_SCRATCH_DIR:?}"
ROOT="$LUMI_PROJECT_DIR/gaitpinn"          # code, container, venv
SCR="$LUMI_SCRATCH_DIR/gaitpinn"           # data, runs, checkpoints, logs
IMAGE="lumi-multitorch-full-u24r72f21m50t211-20261028_090706"
SIF="$ROOT/container/$IMAGE.sif"
RUNS_LOCAL="${RUNS_PATH:-$REPO/../runs}"
# Non-interactive ssh everywhere: fail fast instead of hanging on a prompt.
SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=30)
ssh() { command ssh "${SSH_OPTS[@]}" "$@"; }
export RSYNC_RSH="ssh -o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=30"
