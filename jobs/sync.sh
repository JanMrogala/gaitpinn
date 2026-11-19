#!/bin/bash
# Copy the current commit (clean tree only) to LUMI project storage, with its hash in COMMIT.
source "$(dirname "$0")/env.sh"
cd "$REPO"
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  echo "refusing to sync: uncommitted changes" >&2; exit 1
fi
COMMIT="$(git rev-parse HEAD)"
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
git archive HEAD | tar -x -C "$TMP"
echo "$COMMIT" > "$TMP/COMMIT"
ssh "$LUMI_HOST" "mkdir -p '$ROOT/code'"
rsync -a --delete "$TMP/" "$LUMI_HOST:$ROOT/code/"
echo "synced $COMMIT to $LUMI_HOST:$ROOT/code"
