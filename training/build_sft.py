"""Turn oracle-labeled game states into training rows.

  python training/build_sft.py --data data/v1 --out data/sft --temperature 50

Each row holds the chat messages Ollama would score, the option keys in
letter order, a soft target distribution from the oracle's per-option
values (softmax of value / temperature), and the oracle's best option.
Exact duplicate states are dropped within each split.
"""
import argparse
import glob
import json
import math
import os
import random
from collections import Counter

from prompt import messages


def soft_target(values: dict, keys: list[str], temperature: float) -> list[float]:
    vs = [values[k] for k in keys]
    top = max(vs)
    ws = [math.exp((v - top) / temperature) for v in vs]
    total = sum(ws)
    return [round(w / total, 6) for w in ws]


def build(split_dir: str, temperature: float) -> list[dict]:
    rows, seen = [], set()
    for path in sorted(glob.glob(os.path.join(split_dir, "*", "states.jsonl"))):
        for line in open(path):
            r = json.loads(line)
            if not r.get("values"):
                continue
            keys = list(r["criteria"].keys())
            sig = json.dumps([r["state"], keys], sort_keys=False)
            if sig in seen:
                continue
            seen.add(sig)
            rows.append({
                "messages": messages(r["state"], r["instructions"], r["criteria"]),
                "keys": keys,
                "target": soft_target(r["values"], keys, temperature),
                "label": r["label"],
                "player": r["player"],
                "seed": r["seed"],
            })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/v1")
    ap.add_argument("--out", default="data/sft")
    ap.add_argument("--temperature", type=float, default=50.0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for split in ("train", "val"):
        rows = build(os.path.join(args.data, split), args.temperature)
        random.Random(0).shuffle(rows)
        with open(os.path.join(args.out, f"{split}.jsonl"), "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        peaked = sum(max(r["target"]) > 0.9 for r in rows)
        tied = sum(sorted(r["target"])[-2] > 0.4 for r in rows)
        print(f"{split}: {len(rows)} rows, players {dict(Counter(r['player'] for r in rows))}, "
              f"labels {dict(Counter(r['label'] for r in rows))}, target max>0.9: {peaked}, near-ties: {tied}")


if __name__ == "__main__":
    main()
