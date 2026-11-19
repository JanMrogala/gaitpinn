#!/bin/bash
# One training run on 1 GCD. Submitted by jobs/submit.sh, which sets account, partition,
# time, log paths and the GP_* variables. Arguments are passed to gaitpinn.train.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
set -euo pipefail
module purge >/dev/null 2>&1
module load Local-LAIF lumi-aif-singularity-bindings
export MIOPEN_USER_DB_PATH="/tmp/${USER}-miopen-${SLURM_JOB_ID}"
export MIOPEN_CUSTOM_CACHE_DIR="$MIOPEN_USER_DB_PATH"
mkdir -p "$MIOPEN_USER_DB_PATH"
trap 'rm -rf "$MIOPEN_USER_DB_PATH"' EXIT
cd "$GP_ROOT/code"
singularity exec "$GP_SIF" bash -c \
  "source '$GP_ROOT/venv/bin/activate' && python -m gaitpinn.train --runs '$GP_SCR/runs' $*"
