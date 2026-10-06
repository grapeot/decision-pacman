"""Fold an MLX LoRA adapter (from train_mlx.py) into the Hugging Face checkpoint.

  python training/merge_mlx.py --adapter runs/ft_q1_mlx/adapter --out runs/ft_q1_mlx/merged

The output is the base model's own Hugging Face layout (same files, same
tensor names, bf16), with W + scale * (A @ B)^T on every adapted linear layer,
so llama.cpp's convert_hf_to_gguf.py reads it like the merged checkpoint of
the CUDA path. Only linear weights change, and mlx-lm loads those without any
renaming beyond the `model.language_model` -> `language_model.model` prefix.
"""
import argparse
import json
import os
import shutil

import mlx.core as mx
from huggingface_hub import snapshot_download


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter", required=True, help="directory with adapters.safetensors and adapter_config.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default=None, help="Hugging Face repo or local directory (default: the adapter's base)")
    args = ap.parse_args()

    with open(os.path.join(args.adapter, "adapter_config.json")) as f:
        cfg = json.load(f)
    scale = cfg["lora_parameters"]["scale"]
    base = args.base or cfg["model"]
    src = base if os.path.isdir(base) else snapshot_download(base)
    lora = mx.load(os.path.join(args.adapter, "adapters.safetensors"))
    pending = {k[: -len(".lora_a")] for k in lora if k.endswith(".lora_a")}

    os.makedirs(args.out, exist_ok=True)
    for name in sorted(os.listdir(src)):
        path = os.path.join(src, name)
        if not name.endswith(".safetensors"):
            if os.path.isfile(path):
                shutil.copyfile(path, os.path.join(args.out, name))
            continue
        weights = mx.load(path)
        for key in list(weights):
            if not (key.startswith("model.language_model.") and key.endswith(".weight")):
                continue
            mod = "language_model.model." + key[len("model.language_model."):-len(".weight")]
            if mod not in pending:
                continue
            a, b = lora[mod + ".lora_a"].astype(mx.float32), lora[mod + ".lora_b"].astype(mx.float32)
            w = weights[key]
            weights[key] = (w.astype(mx.float32) + scale * (a @ b).T).astype(w.dtype)
            pending.discard(mod)
        mx.save_safetensors(os.path.join(args.out, name), weights, metadata={"format": "pt"})
    if pending:
        raise SystemExit(f"{len(pending)} adapter modules had no matching weight, e.g. {sorted(pending)[0]}")
    print(f"merged {len({k[:-7] for k in lora if k.endswith('.lora_a')})} LoRA modules (scale {scale}) -> {args.out}")


if __name__ == "__main__":
    main()
