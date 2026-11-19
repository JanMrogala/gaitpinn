#!/bin/bash
# Submit jobs on LUMI. Usage: jobs/submit.sh smoke
source "$(dirname "$0")/env.sh"
what="${1:?usage: submit.sh smoke}"
case "$what" in
  smoke)
    part=dev-g; time=00:10:00
    args="--config configs/base.yaml --smoke --seed 0" ;;
  *) echo "unknown job: $what" >&2; exit 1 ;;
esac
ssh "$LUMI_HOST" "cd '$ROOT/code' && sbatch --parsable --account='$LUMI_PROJECT' \
  --partition=$part --time=$time --job-name=gp-$what \
  --output='$SCR/logs/%x-%j.out' \
  --export=ALL,GP_ROOT='$ROOT',GP_SCR='$SCR',GP_SIF='$SIF' jobs/run.sbatch $args"
