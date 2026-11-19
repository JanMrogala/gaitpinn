#!/bin/bash -l
# One training run on 1 GCD. Submitted by jobs/submit.sh, which sets account, partition,
# time, log paths and the GP_* variables. Arguments are passed to gaitpinn.train unchanged.
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
if [ "$(cat "$CODE/COMMIT")" != "$GP_COMMIT" ]; then echo "COMMIT mismatch in $CODE" >&2; exit 1; fi
cd "$CODE"
# The container sets its own PYTHONPATH (where torch lives); add our code in front of it.
singularity exec "$GP_SIF" bash -c \
  'source "$1/venv/bin/activate"; export PYTHONPATH="$2/src:${PYTHONPATH:-}"; shift 2
   exec python -m gaitpinn.train "$@"' \
  _ "$GP_ROOT" "$CODE" --runs "$GP_SCR/runs" --code "$CODE" "$@"
