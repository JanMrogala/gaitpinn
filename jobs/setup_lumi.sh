#!/bin/bash
# One-off: copy the container image to project storage, check it, and build the venv on top.
source "$(dirname "$0")/env.sh"
SRC_DIR=/appl/local/laifs/containers/lumi-multitorch-u24r72f21m50t211-20261028_090706
SHA=3ebb506cd91e1160744bd9a6933bd918638e6080543eec2090fca821aead0cb6
ssh "$LUMI_HOST" bash -s <<REMOTE
set -euo pipefail
mkdir -p "$ROOT/container" "$SCR"/{data,runs,ckpt,logs,pip-cache}
if [ ! -f "$SIF" ]; then cp "$SRC_DIR/$IMAGE.sif" "$SIF"; fi
echo "$SHA  $SIF" | sha256sum -c -
module purge >/dev/null 2>&1; module load Local-LAIF lumi-aif-singularity-bindings
singularity exec "$SIF" bash -c '
  set -e
  [ -d "$ROOT/venv" ] || python -m venv --system-site-packages "$ROOT/venv"
  source "$ROOT/venv/bin/activate"
  PIP_CACHE_DIR="$SCR/pip-cache" pip install -q --no-deps --no-build-isolation -e "$ROOT/code"
  python -c "import torch, gaitpinn; print(\"torch\", torch.__version__)"'
REMOTE
