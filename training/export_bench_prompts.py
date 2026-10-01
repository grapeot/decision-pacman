"""Export rendered prompts with reference probabilities for on-device benchmarks.

  python training/export_bench_prompts.py --tokenizer runs/ft_v1/merged \
      --reference runs/score_pacman_08b.json --n 20 --out ios/mac_check/bench_prompts.json

Takes the same validation rows as score_ollama.py, renders each with the
model's chat template exactly as served (thinking off), and attaches the
probabilities Ollama returned for them, so a device can check its output.
"""
import argparse
import json
import random

from transformers import AutoTokenizer

from prompt import letters, messages
from score_ollama import load_states


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--reference", required=True, help="score_ollama.py output with per-row details")
    ap.add_argument("--data", default="data/v1/val")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--temperature", type=float, default=50.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    ref = {(d["seed"], d["tick"]): d for d in json.load(open(args.reference))["details"]}
    rows = load_states(args.data, args.temperature)
    random.Random(0).shuffle(rows)
    out = []
    for r in rows[: args.n]:
        keys = list(r["criteria"].keys())
        text = tok.apply_chat_template(messages(r["state"], r["instructions"], r["criteria"]), tokenize=False, add_generation_prompt=True, enable_thinking=False)
        expected = ref[(r["seed"], r["tick"])]["probabilities"]
        out.append({
            "prompt": text,
            "keys": keys,
            "letters": letters(len(keys)),
            "tokens": len(tok(text, add_special_tokens=False)["input_ids"]),
            "expected": [expected[k] for k in keys],
        })
    with open(args.out, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"wrote {len(out)} prompts, tokens {min(o['tokens'] for o in out)}-{max(o['tokens'] for o in out)} -> {args.out}")


if __name__ == "__main__":
    main()
