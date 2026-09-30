"""Score oracle-labeled validation states through a /v1/systemone endpoint.

  python3 training/score_ollama.py --model tev1:0.8b --n 1000 --out runs/score_tev1_08b.json
  TYPESAFE_API_KEY=... python3 training/score_ollama.py --base-url https://api.typesafe.ai --model jev-latest

Sends each state exactly as the game client does and reports agreement with
the oracle's best option: overall, on decisive states (oracle target above
0.9), and the cross-entropy against the oracle's soft target. Standard
library only.
"""
import argparse
import glob
import json
import math
import os
import random
import time
import urllib.request

from build_sft import soft_target


def load_states(data_dir: str, temperature: float) -> list[dict]:
    rows, seen = [], set()
    for path in sorted(glob.glob(os.path.join(data_dir, "*", "states.jsonl"))):
        for line in open(path):
            r = json.loads(line)
            if not r.get("values"):
                continue
            keys = list(r["criteria"].keys())
            sig = json.dumps([r["state"], keys])
            if sig in seen:
                continue
            seen.add(sig)
            r["target"] = soft_target(r["values"], keys, temperature)
            rows.append(r)
    return rows


def ask(base: str, model: str, r: dict) -> dict:
    api_key = os.environ.get("TYPESAFE_API_KEY")
    body = {
        "model": model,
        **({} if api_key else {"keep_alive": -1}),
        "state": r["state"],
        "questions": {"move": {"type": "choice", "instructions": r["instructions"], "criteria": r["criteria"]}},
    }
    headers = {"Content-Type": "application/json", **({"Authorization": f"Bearer {api_key}"} if api_key else {})}
    req = urllib.request.Request(base.rstrip("/") + "/v1/systemone", data=json.dumps(body).encode(), headers=headers)
    t = time.perf_counter()
    out = json.loads(urllib.request.urlopen(req, timeout=120).read())
    move = out["answers"]["move"]
    return {"choice": move["choice"], "probabilities": move["probabilities"], "ms": (time.perf_counter() - t) * 1000}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-url", default="http://localhost:11434")
    ap.add_argument("--data", default="data/v1/val")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--temperature", type=float, default=50.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = load_states(args.data, args.temperature)
    random.Random(0).shuffle(rows)
    rows = rows[: args.n]
    ask(args.base_url, args.model, rows[0])  # warm-up
    n = ok = decisive = decisive_ok = 0
    ce = 0.0
    lat, details = [], []
    for r in rows:
        a = ask(args.base_url, args.model, r)
        keys = list(r["criteria"].keys())
        best = keys[max(range(len(keys)), key=lambda i: r["target"][i])]
        hit = a["choice"] == best
        n += 1
        ok += hit
        if max(r["target"]) > 0.9:
            decisive += 1
            decisive_ok += hit
        ce -= sum(t * math.log(max(a["probabilities"].get(k, 0.0), 1e-9)) for k, t in zip(keys, r["target"]))
        lat.append(a["ms"])
        details.append({"seed": r["seed"], "tick": r["tick"], "best": best, **a})
    lat.sort()
    report = {
        "model": args.model,
        "n": n,
        "acc": round(ok / n, 4),
        "acc_decisive": round(decisive_ok / max(1, decisive), 4),
        "n_decisive": decisive,
        "ce": round(ce / n, 4),
        "latency_p50_ms": round(lat[len(lat) // 2]),
        "latency_p90_ms": round(lat[int(len(lat) * 0.9)]),
    }
    print(json.dumps(report))
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"report": report, "details": details}, f)


if __name__ == "__main__":
    main()
