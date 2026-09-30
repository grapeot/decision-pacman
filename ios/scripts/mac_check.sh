#!/usr/bin/env bash
# Builds and runs the app's inference code on macOS against the bundled benchmark prompts.
#   ios/scripts/mac_check.sh <model.gguf>
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
"$here/scripts/fetch_llama.sh" >/dev/null
fw="$here/Vendor/build-apple/llama.xcframework/macos-arm64_x86_64"
out="$(mktemp -d)/mac_check"
swiftc -O -F "$fw" -framework llama -Xlinker -rpath -Xlinker "$fw" -o "$out" \
  "$here/DecisionPacman/DecisionEngine.swift" "$here/DecisionPacman/Benchmark.swift" "$here/mac_check/main.swift"
"$out" "$1" "$here/DecisionPacman/bench_prompts.json" 2>/dev/null
