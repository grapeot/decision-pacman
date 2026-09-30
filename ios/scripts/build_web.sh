#!/usr/bin/env bash
# Builds the web game and copies it into the app bundle sources (DecisionPacman/Web, not committed).
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
repo="$(cd "$here/.." && pwd)"
(cd "$repo" && npm run build --silent >/dev/null)
rm -rf "$here/DecisionPacman/Web"
cp -R "$repo/dist" "$here/DecisionPacman/Web"
echo "web build copied to DecisionPacman/Web"
