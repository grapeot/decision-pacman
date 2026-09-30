"""Score the same validation states with Hugging Face weights, for parity with Ollama.

  python training/score_hf.py --model runs/ft_v1/merged --n 50 --out runs/ft_v1/score_hf.json

Uses the same rows as score_ollama.py (same shuffle and prefix) and the same
prompt rendering as training, and reads the option-letter probabilities at
the answer position. Runs on CPU unless CUDA is available.
"""
import argparse
import json
import random

import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer

from prompt import letters, messages
from score_ollama import load_states


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default="data/v1/val")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--temperature", type=float, default=50.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.bfloat16).to(device).eval()
    rows = load_states(args.data, args.temperature)
    random.Random(0).shuffle(rows)
    out = []
    for r in rows[: args.n]:
        keys = list(r["criteria"].keys())
        text = tok.apply_chat_template(messages(r["state"], r["instructions"], r["criteria"]), tokenize=False, add_generation_prompt=True, enable_thinking=False)
        ids = tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
        with torch.no_grad():
            logits = model(input_ids=ids, logits_to_keep=1).logits[0, -1].float()
        letter_ids = [tok.encode(c, add_special_tokens=False)[0] for c in letters(len(keys))]
        p = torch.softmax(logits[letter_ids], dim=-1).tolist()
        out.append({"seed": r["seed"], "tick": r["tick"], "probabilities": dict(zip(keys, p)), "tokens": ids.shape[1]})
    with open(args.out, "w") as f:
        json.dump(out, f)
    print(f"scored {len(out)} rows -> {args.out}")


if __name__ == "__main__":
    main()
