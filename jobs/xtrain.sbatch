#!/bin/bash -l
# Study 2 (PLAN2.md) array job on 1 GCD. GP_KIND selects the mapping of the array index i:
#   xtune : i -> fold = i / 24, model = (base, pinn)[(i / 12) % 2], config = i % 12  (96 tasks)
#   xcal  : i -> fold = i                                                          (4 tasks)
#   xfinal: i -> fold = i / 10, model = (base, pinn)[(i / 5) % 2], seed = i % 5     (40 tasks)
# Study 3 (PLAN3.md), share = (0.0, 0.5, 0.9, 0.99)[...]:
#   gmasks: mask statistics and calibration-trial setups                         (1 task)
#   gtune : i -> share i / 128, fold (i / 32) % 4, model (i / 16) % 2, cfg i % 16 (512 tasks)
#   gfinal: i -> share i / 40, fold (i / 10) % 4, model (i / 5) % 2, seed i % 5   (160 tasks)
#   glab  : i -> share 1 + i / 20, fold (i / 5) % 4, model pinn_lab, seed i % 5   (60 tasks)
# Study 4 (PLAN4.md): the same gtune/gfinal with GP_PREFIX=g4 (EMG only) or g4a (EMG + acc);
#   gdistill: i -> share 1 + i / 20, fold (i / 5) % 4, model distill, seed i % 5    (60 tasks)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G
set -euo pipefail
module purge
module load Local-LAIF lumi-aif-singularity-bindings
export MIOPEN_USER_DB_PATH="/tmp/${USER}-miopen-${SLURM_JOB_ID}"
export MIOPEN_CUSTOM_CACHE_DIR="$MIOPEN_USER_DB_PATH"
mkdir -p "$MIOPEN_USER_DB_PATH"
trap 'rm -rf "$MIOPEN_USER_DB_PATH"' EXIT
CODE="$GP_ROOT/code/$GP_COMMIT"
if [ "$(cat "$CODE/COMMIT")" != "$GP_COMMIT" ]; then echo "COMMIT mismatch" >&2; exit 1; fi
MODELS=(base pinn)
SHARES=(0.0 0.5 0.9 0.99)
i=$SLURM_ARRAY_TASK_ID
MODULE=gaitpinn.cross
P="${GP_PREFIX:-g}"
case "$GP_KIND" in g*) MODULE=gaitpinn.gaps ;; esac
case "$GP_KIND" in
  gmasks) args=(gmasks) ;;
  g4pilot) args=(g4pilot --index "$i") ;;
  gtune)  args=(gtune --share "${SHARES[$((i / 128))]}" --fold $(((i / 32) % 4))
                --model "${MODELS[$(((i / 16) % 2))]}" --cfg $((i % 16)) --prefix "$P") ;;
  gfinal) args=(gfinal --share "${SHARES[$((i / 40))]}" --fold $(((i / 10) % 4))
                --model "${MODELS[$(((i / 5) % 2))]}" --seed $((i % 5))
                --chosen "$GP_SCR/chosen_$P.json" --prefix "$P") ;;
  glab)   args=(gfinal --share "${SHARES[$((1 + i / 20))]}" --fold $(((i / 5) % 4))
                --model pinn_lab --seed $((i % 5)) --chosen "$GP_SCR/chosen_$P.json" --prefix "$P") ;;
  gdistill) args=(gfinal --share "${SHARES[$((1 + i / 20))]}" --fold $(((i / 5) % 4))
                --model distill --seed $((i % 5)) --chosen "$GP_SCR/chosen_$P.json" --prefix "$P"
                --teacher-chosen "$GP_SCR/chosen_g.json") ;;
  xtune)  args=(xtune --fold $((i / 24)) --model "${MODELS[$(((i / 12) % 2))]}" --cfg $((i % 12))) ;;
  xcal)   args=(xcal --fold "$i" --chosen "$GP_SCR/chosen_x.json") ;;
  xfinal) args=(xfinal --fold $((i / 10)) --model "${MODELS[$(((i / 5) % 2))]}" --seed $((i % 5))
                --chosen "$GP_SCR/chosen_x.json") ;;
  *) echo "unknown GP_KIND $GP_KIND" >&2; exit 1 ;;
esac
echo "task $i: ${args[*]}"
cd "$CODE"
singularity exec "$GP_SIF" bash -c \
  'source "$1/venv/bin/activate"; export PYTHONPATH="$2/src:${PYTHONPATH:-}"; shift 2
   exec python -m "$@"' \
  _ "$GP_ROOT" "$CODE" "$MODULE" "${args[@]}" --h5 "$GP_SCR/data/h5x" --runs "$GP_SCR/runs" --code "$CODE"
