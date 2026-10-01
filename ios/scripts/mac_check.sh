#!/usr/bin/env bash
# Builds and runs the app's inference engines on macOS against fixed game prompts.
#   ios/scripts/mac_check.sh decision <model.gguf>   fine-tuned decision model (bench_prompts.json)
#   ios/scripts/mac_check.sh chat <model.gguf>       plain chat model under the move grammar (chat_prompts.json)
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
mode="${1:?decision or chat}"
model="${2:?path to a GGUF model}"
case "$mode" in
  decision) prompts="$here/mac_check/bench_prompts.json" ;;
  chat) prompts="$here/mac_check/chat_prompts.json" ;;
  *) echo "mode must be decision or chat" >&2; exit 2 ;;
esac
"$here/scripts/fetch_llama.sh" >/dev/null
fw="$here/Vendor/build-apple/llama.xcframework/macos-arm64_x86_64"
out="$(mktemp -d)/mac_check"
swiftc -O -F "$fw" -framework llama -Xlinker -rpath -Xlinker "$fw" -o "$out" \
  "$here/DecisionPacman/DecisionEngine.swift" "$here/DecisionPacman/ChatMoveEngine.swift" "$here/mac_check/main.swift"
"$out" "$mode" "$model" "$prompts" 2>/dev/null
