#!/usr/bin/env bash
# Publish the demo-only static copy of the web UI to GitHub Pages (a free website).
#
#   scripts/publish_pages.sh
#
# Builds the site with scripts/build_pages.py, then force-pushes it as a fresh single-commit `gh-pages`
# branch (so the repository does not grow with every publish). Turn Pages on once, with the source set to
# the gh-pages branch:   gh api -X POST repos/<owner>/<repo>/pages -f 'source[branch]=gh-pages' -f 'source[path]=/'
# The site is then at https://<owner>.github.io/<repo>/.  Run this again after changing the UI.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
REMOTE_URL="${PAGES_REMOTE_URL:-$(git remote get-url origin)}"   # CI passes a token URL
SHA="$(git rev-parse --short HEAD)"
TMP="$(mktemp -d "${TMPDIR:-/tmp}/hybridock-pages.XXXXXX")"
trap 'rm -rf "$TMP"' EXIT

python3 scripts/build_pages.py --out "$TMP/site"

cd "$TMP/site"
git init -q
git checkout -q -b gh-pages
git add -A
git -c user.name="${GIT_AUTHOR_NAME:-$(git -C "$OLDPWD" config user.name)}" \
    -c user.email="${GIT_AUTHOR_EMAIL:-$(git -C "$OLDPWD" config user.email)}" \
    commit -q -m "Publish the web demo (from ${SHA})"
git push -q --force "$REMOTE_URL" gh-pages
echo "published ${SHA} to the gh-pages branch"   # never print the URL: it may carry a token
