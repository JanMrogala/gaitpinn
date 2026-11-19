#!/bin/bash
# Copy the current commit (clean tree only) to LUMI as code/<commit>/, with its hash in COMMIT.
# Each commit gets its own folder, so a running or queued job never sees its code change.
source "$(dirname "$0")/env.sh"
cd "$REPO"
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  echo "refusing to sync: uncommitted changes" >&2; exit 1
fi
COMMIT="$(git rev-parse HEAD)"
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
chmod 755 "$TMP"
git archive HEAD | tar -x -C "$TMP"
echo "$COMMIT" > "$TMP/COMMIT"
ssh "$LUMI_HOST" "mkdir -p '$ROOT/code/$COMMIT'"
rsync -a --delete "$TMP/" "$LUMI_HOST:$ROOT/code/$COMMIT/"
echo "synced $COMMIT to $LUMI_HOST:$ROOT/code/$COMMIT"
