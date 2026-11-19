#!/bin/bash
# Copy run records back from LUMI (no checkpoints), fill GPU-hours from sacct, rebuild runs.csv.
source "$(dirname "$0")/env.sh"
mkdir -p "$RUNS_LOCAL"
rsync -a --exclude '*.pt' --exclude '*.ckpt' "$LUMI_HOST:$SCR/runs/" "$RUNS_LOCAL/"
jobs=$("$REPO/.venv/bin/python" -m gaitpinn.runs pending --runs "$RUNS_LOCAL")
if [ -n "$jobs" ]; then
  ssh "$LUMI_HOST" "sacct -X -n -P -j $jobs --format=JobIDRaw,State,ElapsedRaw,AllocTRES" \
    > "$RUNS_LOCAL/.sacct.txt"
  "$REPO/.venv/bin/python" -m gaitpinn.runs fill --runs "$RUNS_LOCAL" --sacct "$RUNS_LOCAL/.sacct.txt"
fi
"$REPO/.venv/bin/python" -m gaitpinn.runs index --runs "$RUNS_LOCAL"
