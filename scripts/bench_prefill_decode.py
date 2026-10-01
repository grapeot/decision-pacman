"""Split decision latency into reading the prompt (prefill) and writing tokens (decode).

Sends the rendered game prompts in ios/mac_check/bench_prompts.json, once
with 1 output token (what a decision model needs: one forward pass, then read
the option logits) and once with several (what a chat model writing a small
JSON answer needs). Works against Ollama (/api/generate, raw prompt) or a
llama.cpp server (/completion), so the same GGUF can be timed on different
hardware.

  python3 scripts/bench_prefill_decode.py --backend ollama --model tev1:4b
  python3 scripts/bench_prefill_decode.py --backend llamacpp --base-url http://127.0.0.1:8190 --model tev1-4b

Every request starts with a unique tag, so no request can reuse another's
prompt cache (servers keep several cache slots, so repeated prompts would look
far faster than a real game). Decode cost is reported per output step: the first
token comes out of prefill, and each further token is one decode step.
"""

import argparse
import json
import statistics
import time
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor


def post(url: str, body: dict) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=120).read())


def ask(backend: str, base: str, model: str, prompt: str, n: int) -> dict:
    """Returns wall, prefill and decode milliseconds, and token counts."""
    prompt = prompt.replace("<|im_start|>system\n", f"<|im_start|>system\n[{uuid.uuid4().hex[:12]}] ", 1)
    t = time.perf_counter()
    if backend == "ollama":
        r = post(base + "/api/generate", {"model": model, "prompt": prompt, "raw": True, "stream": False, "keep_alive": -1,
                                         "options": {"num_predict": n, "temperature": 0}})
        out = {"prompt_n": r.get("prompt_eval_count", 0), "prefill_ms": r.get("prompt_eval_duration", 0) / 1e6,
               "decode_n": r.get("eval_count", 0), "decode_ms": r.get("eval_duration", 0) / 1e6}
    else:
        r = post(base + "/completion", {"prompt": prompt, "n_predict": n, "temperature": 0, "cache_prompt": False, "ignore_eos": n > 1})
        tm = r["timings"]
        out = {"prompt_n": tm["prompt_n"], "prefill_ms": tm["prompt_ms"], "decode_n": tm["predicted_n"], "decode_ms": tm["predicted_ms"]}
    out["wall_ms"] = (time.perf_counter() - t) * 1000
    return out


def summarize(rows: list[dict]) -> dict:
    med = lambda k: round(statistics.median(r[k] for r in rows), 1)
    per_token = [r["prefill_ms"] / r["prompt_n"] for r in rows if r["prompt_n"]]
    steps = [r["decode_ms"] / (r["decode_n"] - 1) for r in rows if r["decode_n"] > 1]
    return {"n": len(rows), "wall_p50": med("wall_ms"), "prefill_p50": med("prefill_ms"), "prompt_tokens_p50": med("prompt_n"),
            "prefill_ms_per_token": round(statistics.median(per_token), 3) if per_token else None,
            "decode_p50": med("decode_ms"), "decode_tokens_p50": med("decode_n"),
            "ms_per_decode_step": round(statistics.median(steps), 2) if steps else None}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=["ollama", "llamacpp"], required=True)
    ap.add_argument("--base-url", default="http://localhost:11434")
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompts", default="ios/mac_check/bench_prompts.json")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--decode", type=int, nargs="+", default=[1, 7], help="output token counts to time")
    ap.add_argument("--concurrency", type=int, default=1, help="parallel requests for a throughput run (1-token answers)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    prompts = [p["prompt"] for p in json.load(open(args.prompts))]
    base = args.base_url.rstrip("/")
    ask(args.backend, base, args.model, prompts[0], 1)  # load the model
    report = {"backend": args.backend, "model": args.model}
    for n in args.decode:
        rows = [ask(args.backend, base, args.model, p, n) for _ in range(args.rounds) for p in prompts]
        report[f"out_{n}"] = summarize(rows)
        print(f"{n} output token(s): {report[f'out_{n}']}", flush=True)
    if args.concurrency > 1:
        work = prompts * args.rounds
        t = time.perf_counter()
        with ThreadPoolExecutor(args.concurrency) as pool:
            rows = list(pool.map(lambda p: ask(args.backend, base, args.model, p, 1), work))
        secs = time.perf_counter() - t
        report[f"concurrency_{args.concurrency}"] = {**summarize(rows), "decisions_per_s": round(len(work) / secs, 1)}
        print(f"concurrency {args.concurrency}: {report[f'concurrency_{args.concurrency}']}", flush=True)
    if args.out:
        json.dump(report, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
