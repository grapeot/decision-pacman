#!/usr/bin/env bash
# Merge a LoRA adapter and convert it to one GGUF file for Ollama.
#   training/export_gguf.sh <adapter_dir> <out.gguf> <llama.cpp_dir> [outtype=q8_0]
# The merge runs in the current Python (the training environment); the
# conversion uses llama.cpp's own environment at <llama.cpp_dir>/.venv.
set -euo pipefail
adapter=$1; out=$2; llama=$3; outtype=${4:-q8_0}
here="$(cd "$(dirname "$0")" && pwd)"
merged="$(dirname "$adapter")/merged"
python "$here/merge.py" --adapter "$adapter" --out "$merged"
mkdir -p "$(dirname "$out")"
"$llama/.venv/bin/python" "$llama/convert_hf_to_gguf.py" "$merged" --outtype "$outtype" --outfile "$out"
ls -la "$out"
