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
  final-strong)
    # strongly physics-constrained pinn: tuned pinn settings with a fixed lambda (PLAN.md log)
    python3 -c "import json,sys; c=json.load(open(sys.argv[1])); c['lam']=0.1; json.dump(c, open(sys.argv[2],'w'), indent=2)" \
      "$RUNS_LOCAL/chosen_pinn.json" "$RUNS_LOCAL/chosen_pinn_strong.json"
    rsync -a "$RUNS_LOCAL/chosen_pinn_strong.json" "$LUMI_HOST:$SCR/"
    n=$(ssh "$LUMI_HOST" "ls '$SCR/data/h5'/AB*.h5 | wc -l")
    part=small-g; time="${GP_TIME:-03:00:00}"; script="--array=0-$((n - 1)) jobs/train.sbatch"
    args=""; extra=",GP_KIND=final,GP_MODELS=pinn_strong" ;;
  tune|final)
    if [ "$what" = final ]; then  # the configurations chosen on walking validation
      for m in base pinn; do
        test -f "$RUNS_LOCAL/chosen_$m.json" || { echo "missing chosen_$m.json" >&2; exit 1; }
      done
      rsync -a "$RUNS_LOCAL"/chosen_base.json "$RUNS_LOCAL"/chosen_pinn.json "$LUMI_HOST:$SCR/"
    fi
    # 2 models x number of people, one array task each
    n=$(ssh "$LUMI_HOST" "ls '$SCR/data/h5'/AB*.h5 | wc -l")
    part=small-g; time="${GP_TIME:-02:00:00}"; script="--array=0-$((2 * n - 1)) jobs/train.sbatch"
    args=""; extra=",GP_KIND=$what" ;;
  *) echo "unknown job: $what" >&2; exit 1 ;;
esac
ssh "$LUMI_HOST" "test -f '$ROOT/code/$COMMIT/COMMIT' || { echo 'commit not synced' >&2; exit 1; }
  cd '$ROOT/code/$COMMIT' && sbatch --parsable --account='$LUMI_PROJECT' \
  --partition=$part --time=$time --job-name=gp-$what \
  --output='$SCR/logs/%x-%A_%a.out' \
  --export=ALL,GP_ROOT='$ROOT',GP_SCR='$SCR',GP_SIF='$SIF',GP_COMMIT='$COMMIT'${extra:-} \
  $script $args"
