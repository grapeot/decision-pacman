"""Score validation states with MLX weights, for parity with Ollama (the MLX twin of score_hf.py).

  python training/score_mlx.py --model runs/ft_q1_mlx/merged --data data/q1/val --target label --n 50 --out runs/ft_q1_mlx/score_mlx.json
  python training/score_mlx.py --model Qwen/Qwen3.5-0.8B --adapter runs/ft_q1_mlx/adapter --data data/q1/val --target label --n 50 --out ...

Uses the same rows as score_ollama.py (same shuffle and prefix) and the same
prompt rendering as training, and reads the option-letter probabilities at
the answer position. Writes the same format as score_hf.py.
"""
import argparse
import json
import random

import mlx.core as mx
from mlx_lm import load

from prompt import letters, messages
from score_ollama import load_states


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="merged checkpoint, or the base model when --adapter is given")
    ap.add_argument("--adapter", default=None, help="adapter directory written by train_mlx.py")
    ap.add_argument("--data", default="data/v1/val")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--temperature", type=float, default=50.0)
    ap.add_argument("--target", choices=["values", "label"], default="values")
    ap.add_argument("--encoder", default=None, help="score this encoding from the record's alt field")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    model, tok = load(args.model, adapter_path=args.adapter)
    model.eval()
    rows = load_states(args.data, args.temperature, args.target, args.encoder)
    random.Random(0).shuffle(rows)
    out = []
    for r in rows[: args.n]:
        keys = list(r["criteria"].keys())
        text = tok.apply_chat_template(messages(r["state"], r["instructions"], r["criteria"]), tokenize=False, add_generation_prompt=True, enable_thinking=False)
        ids = tok.encode(text, add_special_tokens=False)
        logits = model(mx.array([ids]))[0, -1].astype(mx.float32)
        letter_ids = [tok.encode(c, add_special_tokens=False)[0] for c in letters(len(keys))]
        p = mx.softmax(logits[mx.array(letter_ids)], axis=-1).tolist()
        out.append({"seed": r["seed"], "tick": r["tick"], "probabilities": dict(zip(keys, p)), "tokens": len(ids)})
    with open(args.out, "w") as f:
        json.dump(out, f)
    print(f"scored {len(out)} rows -> {args.out}")


if __name__ == "__main__":
    main()
