#!/bin/bash
# Publish the built site (../site) to the gh-pages branch, which holds ONLY the site.
# Run `make site` first. Never publishes code, data or anything outside ../site.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
SITE="$REPO/../site"
[ -f "$SITE/index.html" ] || { echo "build the site first: make site" >&2; exit 1; }
if grep -rqE "/home/|\.env\.local|project_[0-9]{9}" --include='*.html' "$SITE" \
   || grep -rqE "/home/|project_[0-9]{9}" "$SITE"; then echo "site contains local paths or IDs" >&2; exit 1; fi
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
cp -r "$SITE/." "$TMP/"
cd "$TMP"
git init -q -b gh-pages
git add -A
git commit -q -m "Built site from runs/ (gaitpinn.site, code $(git -C "$REPO" rev-parse --short HEAD))"
git push -q -f "$(git -C "$REPO" remote get-url origin)" gh-pages
echo "published to gh-pages"
