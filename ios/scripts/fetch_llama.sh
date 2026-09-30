#!/usr/bin/env bash
# Downloads the pinned llama.cpp xcframework into ios/Vendor (not committed).
set -euo pipefail
TAG=${LLAMA_TAG:-b11298}
cd "$(dirname "$0")/../Vendor" 2>/dev/null || { mkdir -p "$(dirname "$0")/../Vendor"; cd "$(dirname "$0")/../Vendor"; }
if [[ -d build-apple/llama.xcframework && -f .tag && "$(cat .tag)" == "$TAG" ]]; then echo "llama.xcframework $TAG present"; exit 0; fi
curl -fsSL -o "llama-$TAG-xcframework.zip" "https://github.com/ggml-org/llama.cpp/releases/download/$TAG/llama-$TAG-xcframework.zip"
rm -rf build-apple && unzip -q "llama-$TAG-xcframework.zip" && rm "llama-$TAG-xcframework.zip"
echo "$TAG" > .tag
echo "llama.xcframework $TAG ready"
