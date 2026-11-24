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
  prepare-x)  # study 2: also convert ramp files, into data/h5x
    part=dev-g; time=01:00:00; script=jobs/prepare.sbatch; args=""; extra=",GP_H5=h5x,GP_RAMPS=1" ;;
  xtune|xcal|xfinal)  # study 2 (PLAN2.md)
    case "$what" in xtune) n=96 ;; xcal) n=4 ;; xfinal) n=40 ;; esac
    if [ "$what" != xtune ]; then
      test -f "$RUNS_LOCAL/chosen_x.json" || { echo "run: python -m gaitpinn.test xselect" >&2; exit 1; }
      rsync -a "$RUNS_LOCAL/chosen_x.json" "$LUMI_HOST:$SCR/"
    fi
    part=small-g; time="${GP_TIME:-04:00:00}"
    script="--array=${GP_ARRAY:-0-$((n - 1))} jobs/xtrain.sbatch"  # GP_ARRAY: e.g. a pilot subset
    args=""; extra=",GP_KIND=$what" ;;
  gmasks|gtune|gfinal|glab|gdistill|g4pilot)  # studies 3 and 4 (PLAN3.md, PLAN4.md)
    # GP_PREFIX: g (study 3, default), g4 (study 4, EMG only), g4a (study 4, EMG + acc)
    P="${GP_PREFIX:-g}"
    case "$what" in
      gmasks) n=1 ;; gtune) n=512 ;; gfinal) n=160 ;; glab|gdistill) n=60 ;; g4pilot) n=20 ;;
    esac
    if [ "$what" = gfinal ] || [ "$what" = glab ] || [ "$what" = gdistill ]; then
      test -f "$RUNS_LOCAL/chosen_$P.json" || { echo "run: gaitpinn.gaps gselect --prefix $P" >&2; exit 1; }
      rsync -a "$RUNS_LOCAL/chosen_$P.json" "$LUMI_HOST:$SCR/"
      if [ "$what" = gdistill ]; then
        [ "$P" = g ] && { echo "gdistill needs GP_PREFIX=g4 or g4a" >&2; exit 1; }
        rsync -a "$RUNS_LOCAL/chosen_g.json" "$LUMI_HOST:$SCR/"
      fi
    fi
    part=small-g; time="${GP_TIME:-02:00:00}"
    script="--array=${GP_ARRAY:-0-$((n - 1))} jobs/xtrain.sbatch"
    args=""; extra=",GP_KIND=$what,GP_PREFIX=$P"; jobname="gp-$P-$what" ;;
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
  --partition=$part --time=$time --job-name=${jobname:-gp-$what} \
  --output='$SCR/logs/%x-%A_%a.out' \
  --export=ALL,GP_ROOT='$ROOT',GP_SCR='$SCR',GP_SIF='$SIF',GP_COMMIT='$COMMIT'${extra:-} \
  $script $args"
