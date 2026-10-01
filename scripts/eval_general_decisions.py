"""Score decision models on non-Pac-Man decisions: the public JevBench items.

JevBench (MIT, github.com/fstandhartinger/jevbench) publishes labeled decisions
in TypeSafe's /v1/systemone format: yes/no (`noul`), pick-one (`choice`), and
ordinal (`score`) questions over short texts such as refund policies, support
messages, and tool requests. This script downloads the three public splits from
a pinned commit, checks their hashes, and sends every item that fits the
model's context to /v1/systemone, one request at a time.

Each item is scored by argmax against its label, as JevBench does, and by the
probability the model gave the correct option. Items are not repeated, so no
timed request hits the server's response cache.

Usage: python3 scripts/eval_general_decisions.py [--out runs/x.jsonl] model [model ...]
"""
import argparse, hashlib, json, os, statistics, subprocess, time, urllib.error, urllib.request

COMMIT = "bb05a335bc809e61b20c0f745d25499a82b326fc"
RAW = f"https://raw.githubusercontent.com/fstandhartinger/jevbench/{COMMIT}/datasets/public/"
SPLITS = {  # sha256 values as listed in the repository's datasets/manifest.json
    "easy": "231df3c2c8e88a1a8c137ebe85de96ba70fabd330849098ac7b3c52c70b7172b",
    "original": "5c2414edb3006b8bfcb70fda433f0f9ca015759433849f8d3104328a1f7c4180",
    "hard": "89e9e6becb33ed88c1de7d42dcc87531b2fb64cfaef4e1986faf7c37b3f80ebb",
}
WARMUP = {"state": "The sky is green today.",
          "questions": {"w": {"type": "noul", "instructions": "Does the text say the sky is blue?"}}}


def load_split(name, cache_dir):
    path = os.path.join(cache_dir, f"{name}.jsonl")
    if not os.path.exists(path):
        os.makedirs(cache_dir, exist_ok=True)
        data = urllib.request.urlopen(RAW + f"{name}.jsonl", timeout=60).read()
        with open(path, "wb") as f:
            f.write(data)
    data = open(path, "rb").read()
    if hashlib.sha256(data).hexdigest() != SPLITS[name]:
        raise SystemExit(f"{path}: sha256 mismatch, delete it and rerun")
    return [json.loads(line) for line in data.decode().splitlines() if line.strip()]


def request_chars(item):
    state = item["state"] if isinstance(item["state"], str) else json.dumps(item["state"])
    return len(state) + len(json.dumps(item["question"]))


def wait_while(pattern):
    """Do not compete with another process (for example a realtime game) for the server."""
    while pattern and subprocess.run(["pgrep", "-f", pattern], capture_output=True).returncode == 0:
        time.sleep(10)


def post(base, body):
    req = urllib.request.Request(base + "/v1/systemone", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t = time.perf_counter()
    r = json.loads(urllib.request.urlopen(req, timeout=300).read())
    return r, (time.perf_counter() - t) * 1000


def probabilities(item, ans):
    """Map an answer to probabilities over the item's exact labels."""
    if item["question"]["type"] == "noul":
        return {"yes": ans["noul"], "no": 1 - ans["noul"]}
    return ans["probabilities"]


def score_item(base, model, item):
    body = {"model": model, "keep_alive": -1, "state": item["state"],
            "questions": {"decision": item["question"]}}
    rec = {"model": model, "id": item["id"], "family": item["family"], "type": item["question"]["type"],
           "n_options": len(item["labels"]), "expected": str(item["expected"])}
    try:
        r, ms = post(base, body)
        p = probabilities(item, r["answers"]["decision"])
        pred = max(sorted(p), key=lambda k: p[k])
        rec.update(ok=True, latency_ms=ms, input_tokens=r.get("usage", {}).get("input_tokens", 0),
                   predicted=pred, correct=pred == rec["expected"], p_correct=p.get(rec["expected"], 0.0))
    except (urllib.error.URLError, KeyError, ValueError) as e:
        rec.update(ok=False, error=str(e)[:200], correct=False, p_correct=0.0)
    return rec


def summarize(recs):
    ok = [r for r in recs if r["ok"]]
    lat = sorted(r["latency_ms"] for r in ok)
    n = len(recs)
    return {"n": n, "failed": n - len(ok),
            "accuracy": sum(r["correct"] for r in recs) / n,
            "chance": sum(1 / r["n_options"] for r in recs) / n,
            "p_correct": sum(r["p_correct"] for r in recs) / n,
            "p50_ms": statistics.median(lat) if lat else 0,
            "max_tokens": max((r["input_tokens"] for r in ok), default=0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:11434")
    ap.add_argument("--splits", default="easy,original,hard")
    ap.add_argument("--max-request-chars", type=int, default=4000,
                    help="skip items longer than this (state plus question JSON) so they fit a 2,048-token context")
    ap.add_argument("--cache-dir", default="data/jevbench")
    ap.add_argument("--pause-while", default="", help="pgrep -f pattern; wait before each request while it matches")
    ap.add_argument("--out", help="write one JSON record per item and model")
    ap.add_argument("models", nargs="+")
    a = ap.parse_args()

    items, skipped = [], 0
    for split in a.splits.split(","):
        for it in load_split(split, a.cache_dir):
            if request_chars(it) > a.max_request_chars:
                skipped += 1
                continue
            it["split"] = split
            items.append(it)
    print(f"{len(items)} items, {skipped} skipped as too long for the context")

    out = open(a.out, "w") if a.out else None
    print(f"{'model':16s} {'split':9s} {'n':>4s} {'fail':>4s} {'acc':>5s} {'chance':>6s} {'p_corr':>6s} {'p50ms':>6s} {'maxtok':>6s}")
    for model in a.models:
        wait_while(a.pause_while)
        post(a.base_url, {"model": model, "keep_alive": -1, **WARMUP})  # load and warm up; not timed
        recs = []
        for it in items:
            wait_while(a.pause_while)
            rec = score_item(a.base_url, model, it)
            rec["split"] = it["split"]
            recs.append(rec)
            if out:
                out.write(json.dumps(rec) + "\n")
                out.flush()
        groups = [(s, [r for r in recs if r["split"] == s]) for s in a.splits.split(",")] + [("all", recs)]
        for name, group in groups:
            if not group:
                continue
            s = summarize(group)
            print(f"{model:16s} {name:9s} {s['n']:4d} {s['failed']:4d} {s['accuracy']:5.2f} {s['chance']:6.2f} "
                  f"{s['p_correct']:6.2f} {s['p50_ms']:6.0f} {s['max_tokens']:6d}", flush=True)


if __name__ == "__main__":
    main()
