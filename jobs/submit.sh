#!/bin/bash
# Submit jobs on LUMI for the current commit (run jobs/sync.sh first). Usage: jobs/submit.sh smoke
source "$(dirname "$0")/env.sh"
what="${1:?usage: submit.sh smoke}"
COMMIT="$(git -C "$REPO" rev-parse HEAD)"
case "$what" in
  smoke)
    part=dev-g; time=00:10:00
    script=jobs/run.sbatch; args="--config configs/base.yaml --smoke --seed 0" ;;
  prepare)
    part=dev-g; time=01:00:00; script=jobs/prepare.sbatch; args="" ;;
  *) echo "unknown job: $what" >&2; exit 1 ;;
esac
ssh "$LUMI_HOST" "test -f '$ROOT/code/$COMMIT/COMMIT' || { echo 'commit not synced' >&2; exit 1; }
  cd '$ROOT/code/$COMMIT' && sbatch --parsable --account='$LUMI_PROJECT' \
  --partition=$part --time=$time --job-name=gp-$what \
  --output='$SCR/logs/%x-%j.out' \
  --export=ALL,GP_ROOT='$ROOT',GP_SCR='$SCR',GP_SIF='$SIF',GP_COMMIT='$COMMIT' \
  $script $args"
