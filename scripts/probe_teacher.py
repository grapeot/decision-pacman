"""Probe an OpenAI-compatible chat model as a teacher: latency, output-format consistency, accuracy.

Reuses bench.scenario (known correct answer). Modes:
  nothink  - plain prompt asking for JSON, thinking disabled
  schema   - thinking disabled + response_format json_schema (enum of legal moves)
  think    - thinking enabled, JSON requested at the end
Usage: python3 teacher.py MODE [--n N] [--concurrency C]
"""
import argparse, json, random, re, statistics, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
import os
from probe_decision_models import scenario, INSTR

BASE = os.environ.get("TEACHER_BASE_URL", "http://localhost:8000/v1")  # any OpenAI-compatible server
MODEL = os.environ.get("TEACHER_MODEL", "replace-with-your-model-id")


def build(state, options, mode):
    prompt = (f"{INSTR}\n\nState (JSON):\n{json.dumps(state)}\n\nLegal moves: {', '.join(options)}.\n"
              'Answer with only a JSON object: {"move": "<one legal move>", "reason": "<one short sentence>"}')
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}],
            "temperature": 0, "max_tokens": 4096 if mode == "think" else 200,
            "chat_template_kwargs": {"enable_thinking": mode == "think"}}
    if mode == "schema":
        body["response_format"] = {"type": "json_schema", "json_schema": {"name": "move", "schema": {
            "type": "object", "properties": {"move": {"type": "string", "enum": options},
                                             "reason": {"type": "string"}},
            "required": ["move", "reason"], "additionalProperties": False}}}
    return body


def call(args):
    state, options, good, mode = args
    req = urllib.request.Request(BASE + "/chat/completions", data=json.dumps(build(state, options, mode)).encode(),
                                 headers={"Content-Type": "application/json"})
    t = time.perf_counter()
    r = json.loads(urllib.request.urlopen(req, timeout=300).read())
    ms = (time.perf_counter() - t) * 1000
    text = r["choices"][0]["message"].get("content") or ""
    move = None
    m = re.search(r"\{[^{}]*\"move\"[^{}]*\}", text, re.S)
    if m:
        try:
            move = json.loads(m.group(0)).get("move")
        except json.JSONDecodeError:
            pass
    return {"ms": ms, "parsed": move is not None, "legal": move in options, "ok": move == good,
            "fright": state["ghosts_frightened"], "out_tok": r["usage"]["completion_tokens"],
            "in_tok": r["usage"]["prompt_tokens"], "text": text[-200:]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["nothink", "schema", "think"])
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--concurrency", type=int, default=1)
    a = ap.parse_args()
    rng = random.Random(42)
    jobs = [(*scenario(rng), a.mode) for _ in range(a.n)]
    t = time.perf_counter()
    with ThreadPoolExecutor(a.concurrency) as ex:
        res = list(ex.map(call, jobs))
    wall = time.perf_counter() - t
    lat = sorted(r["ms"] for r in res)
    nf = sum(r["fright"] for r in res)
    print(f"mode={a.mode} n={a.n} conc={a.concurrency} p50={statistics.median(lat):.0f}ms "
          f"p90={lat[int(len(lat)*0.9)]:.0f}ms throughput={a.n/wall:.2f}/s "
          f"parsed={sum(r['parsed'] for r in res)}/{a.n} legal={sum(r['legal'] for r in res)}/{a.n} "
          f"acc={sum(r['ok'] for r in res)/a.n:.2f} fright={sum(r['ok'] for r in res if r['fright'])}/{nf} "
          f"in_tok~{res[0]['in_tok']} out_tok_med={statistics.median(r['out_tok'] for r in res):.0f}")
    bad = [r for r in res if not r["ok"]][:3]
    for r in bad:
        print("  miss:", r["text"].replace("\n", " "))


if __name__ == "__main__":
    main()
