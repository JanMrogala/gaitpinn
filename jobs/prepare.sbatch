#!/bin/bash -l
# Unzip the needed Camargo folders and convert all people to HDF5 (8 in parallel).
# Submitted by jobs/submit.sh prepare. Uses 1 GCD slot (8 cores) on a GPU partition because
# CPU partitions are not used in this study.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
set -euo pipefail
module purge
module load Local-LAIF lumi-aif-singularity-bindings
RAW="$GP_SCR/data/raw"; EXT="$GP_SCR/data/extract"; OUT="$GP_SCR/data/h5"
mkdir -p "$EXT" "$OUT"
for z in "$RAW"/Subjects_Part*.zip; do
  for mode in treadmill levelground stair; do
    for kind in ik imu emg conditions; do
      unzip -o -q "$z" "*/$mode/$kind/*.mat" -d "$EXT" < /dev/null || true
    done
  done
done
people=$(cd "$EXT" && ls -d AB* | tr '\n' ' ')
echo "people: $people"
CODE="$GP_ROOT/code/$GP_COMMIT"
singularity exec "$GP_SIF" bash -c \
  'export PYTHONPATH="$1/src:${PYTHONPATH:-}"; shift
   printf "%s\n" $4 | xargs -P 8 -I{} python -m gaitpinn.data --extract "$1" --info "$2" --out "$3" {}' \
  _ "$CODE" "$EXT" "$RAW/SubjectInfo.mat" "$OUT" "$people"
ls -la "$OUT"
