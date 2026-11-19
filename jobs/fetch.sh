#!/bin/bash
# Copy run records back from LUMI (no checkpoints), update the GPU-hour ledger from sacct for
# every gp-* job (failed ones included), fill gpu_hours in run.json, rebuild runs.csv.
source "$(dirname "$0")/env.sh"
PY="$REPO/.venv/bin/python"
mkdir -p "$RUNS_LOCAL"
rsync -a --timeout=120 --exclude '*.pt' --exclude '*.ckpt' "$LUMI_HOST:$SCR/runs/" "$RUNS_LOCAL/"
ssh "$LUMI_HOST" "sacct -X -n -P -S ${GP_SINCE:-2026-11-18} --format=JobIDRaw,JobName,State,ElapsedRaw,AllocTRES" \
  > "$RUNS_LOCAL/.sacct.txt"
"$PY" -m gaitpinn.runs ledger --runs "$RUNS_LOCAL" --sacct "$RUNS_LOCAL/.sacct.txt"
"$PY" -m gaitpinn.runs index --runs "$RUNS_LOCAL"
