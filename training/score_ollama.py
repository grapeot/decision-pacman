"""Score labeled validation states through a /v1/systemone endpoint.

  python3 training/score_ollama.py --model tev1:0.8b --n 1000 --out runs/score_tev1_08b.json
  TYPESAFE_API_KEY=... python3 training/score_ollama.py --base-url https://api.typesafe.ai --model jev-latest
  python3 training/score_ollama.py --chat --model qwen3.5:4b   # a plain chat model, as the llm:<model> policy asks it
  python3 training/score_ollama.py --model pacman-0.8b-qwen --data data/q1/val --target label
  python3 training/score_ollama.py --model pacman-0.8b-qwen-peek --data data/q1/val --target label --encoder features-peek5s

Sends each state exactly as the game client does and reports agreement with
the labeler's best option: overall, on decisive states (target above 0.9),
and the cross-entropy against the target. With --target values the target is
the oracle's soft target; with --target label it is the record's label (for a
chat-model teacher, its move), so every state counts as decisive. --encoder
sends an encoding stored under "alt" (see build_sft.py) instead of the
record's own. With --chat, it asks a plain chat model through /api/chat the
way `src/agent/llm.ts` does: the same prompt, a JSON schema restricted to the
options, thinking off, and probabilities from the logprobs of the move token.
Standard library only.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
import urllib.request

from build_sft import load_records


def load_states(data_dir: str, temperature: float, kind: str = "values", encoder: str | None = None) -> list[dict]:
    """Deduplicated labeled states with a `target`; hard labels are scored unsmoothed."""
    return load_records(data_dir, kind, encoder, temperature, 0.0)


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


LLM_ANSWER = 'Answer with only a JSON object: {"move": "<one of the options>"}'


def chat_prompt(r: dict) -> str:
    """Same text as teacherPrompt(enc, undefined, LLM_ANSWER) in src/agent/teacher.ts."""
    state = r["state"] if isinstance(r["state"], str) else json.dumps(r["state"], separators=(",", ":"), ensure_ascii=False)
    options = "\n".join(f"- {k}: {c}" if c else f"- {k}" for k, c in r["criteria"].items())
    return f"{r['instructions']}\n\nState:\n{state}\n\nOptions:\n{options}\n\n{LLM_ANSWER}"


def move_probabilities(content: str, logprobs: list, keys: list[str]) -> dict:
    """Mass of each option at the token that starts the move value, renormalized (as in src/agent/llm.ts)."""
    colon = content.find(":", content.find('"move"'))
    value_start = content.find('"', colon) + 1
    end, at = 0, None
    for t in logprobs or []:
        end += len(t["token"])
        if end > value_start:
            at = t
            break
    mass = {k: 0.0 for k in keys}
    for cand in (at or {}).get("top_logprobs") or ([at] if at else []):
        text = cand["token"].lstrip(' "\t\n').lower()
        if text:
            for k in keys:
                if k.startswith(text):
                    mass[k] += math.exp(cand["logprob"])
    total = sum(mass.values())
    return {k: v / total for k, v in mass.items()} if total > 0 else {}


def ask_chat(base: str, model: str, r: dict) -> dict:
    keys = list(r["criteria"].keys())
    body = {
        "model": model,
        "messages": [{"role": "user", "content": chat_prompt(r)}],
        "format": {"type": "object", "properties": {"move": {"type": "string", "enum": keys}}, "required": ["move"]},
        "think": False,
        "stream": False,
        "keep_alive": -1,
        "logprobs": True,
        "top_logprobs": 10,
        "options": {"temperature": 0, "num_predict": 32},
    }
    req = urllib.request.Request(base.rstrip("/") + "/api/chat", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t = time.perf_counter()
    out = json.loads(urllib.request.urlopen(req, timeout=120).read())
    ms = (time.perf_counter() - t) * 1000
    content = out["message"]["content"]
    choice = str(json.loads(content).get("move", "")).strip().lower()
    return {"choice": choice, "probabilities": move_probabilities(content, out.get("logprobs"), keys), "ms": ms}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-url", default="http://localhost:11434")
    ap.add_argument("--data", default="data/v1/val")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--temperature", type=float, default=50.0)
    ap.add_argument("--target", choices=["values", "label"], default="values")
    ap.add_argument("--encoder", default=None, help="send this encoding from the record's alt field")
    ap.add_argument("--out", default=None)
    ap.add_argument("--chat", action="store_true", help="ask a plain chat model through /api/chat with constrained JSON")
    args = ap.parse_args()
    ask_fn = ask_chat if args.chat else ask

    rows = load_states(args.data, args.temperature, args.target, args.encoder)
    random.Random(0).shuffle(rows)
    rows = rows[: args.n]
    ask_fn(args.base_url, args.model, rows[0])  # warm-up
    n = ok = decisive = decisive_ok = 0
    ce = 0.0
    lat, details = [], []
    for r in rows:
        a = ask_fn(args.base_url, args.model, r)
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
        **({"chat": True} if args.chat else {}),
        "data": args.data,
        "target": args.target,
        **({"encoder": args.encoder} if args.encoder else {}),
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
