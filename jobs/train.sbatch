#!/bin/bash -l
# Array job: task i trains model MODELS[i % 2] for person PEOPLE[i / 2] on 1 GCD.
#   tune : all 12 grid configurations, seed 100 (walking validation only)
#   final: the chosen configuration, seeds 0-4, plus the test stream
# Submitted by jobs/submit.sh tune|final with GP_KIND set.
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
mapfile -t PEOPLE < <(cd "$GP_SCR/data/h5" && ls AB*.h5 | sed 's/\.h5$//' | sort)
i=$SLURM_ARRAY_TASK_ID
model=${MODELS[$((i % 2))]}
person=${PEOPLE[$((i / 2))]}
echo "task $i: $GP_KIND $model $person"
args=(--config configs/$model.yaml --person "$person" --h5 "$GP_SCR/data/h5" --kind "$GP_KIND")
if [ "$GP_KIND" = final ]; then
  args+=(--chosen "$GP_SCR/chosen_$model.json" --seeds 0 1 2 3 4)
fi
cd "$CODE"
singularity exec "$GP_SIF" bash -c \
  'source "$1/venv/bin/activate"; export PYTHONPATH="$2/src:${PYTHONPATH:-}"; shift 2
   exec python -m gaitpinn.train "$@"' \
  _ "$GP_ROOT" "$CODE" --runs "$GP_SCR/runs" --code "$CODE" "${args[@]}"
