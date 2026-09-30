"""Benchmark decision models on Pac-Man-like states via /v1/systemone.

Every request uses a freshly randomized state so no request hits a response cache.
Usage: python3 bench.py [--base-url URL] [--n N] model [model ...]
"""
import argparse, json, random, statistics, time, urllib.request

INSTR = ("You are Pac-Man. Eat pellets for points. Touching a normal ghost costs a life, so move away "
         "from nearby normal ghosts. Frightened ghosts are harmless and can be eaten for points. "
         "Which direction should Pac-Man take?")
DIRS = ["up", "down", "left", "right"]


def scenario(rng):
    """One clearly good exit and 1-3 clearly bad ones. Returns (state, options, answer)."""
    exits = rng.sample(DIRS, rng.randint(2, 4))
    good = rng.choice(exits)
    frightened = rng.random() < 0.25
    state = {"lives": rng.randint(1, 3), "pellets_left": rng.randint(20, 240),
             "ghosts_frightened": frightened, "exits": {}}
    for d in exits:
        if d == good:
            if frightened:
                e = {"ghost_dist": rng.randint(1, 3), "pellets_5": rng.randint(1, 5)}
            else:
                e = {"ghost_dist": rng.choice([None, rng.randint(8, 20)]), "pellets_5": rng.randint(3, 5)}
        else:
            if frightened:
                e = {"ghost_dist": None, "pellets_5": 0}
            else:
                e = {"ghost_dist": rng.randint(1, 3), "pellets_5": rng.randint(0, 5)}
        e["dead_end"] = False
        state["exits"][d] = e
    return state, exits, good


def to_text(state):
    parts = []
    for d, e in state["exits"].items():
        g = "no ghost that way" if e["ghost_dist"] is None else f"a ghost {e['ghost_dist']} steps away"
        parts.append(f"Going {d}: {g}, {e['pellets_5']} pellets in the next 5 steps.")
    mode = "Ghosts are frightened." if state["ghosts_frightened"] else "Ghosts are dangerous."
    return mode + " " + " ".join(parts)


STYLE = "json"


def ask(base, model, state, options):
    if STYLE == "text":
        state = to_text(state)
    body = {"model": model, "keep_alive": -1, "state": state,
            "questions": {"move": {"type": "choice", "instructions": INSTR,
                                   "criteria": {o: None for o in options}}}}
    req = urllib.request.Request(base + "/v1/systemone", data=json.dumps(body, separators=(",", ":")).encode(),
                                 headers={"Content-Type": "application/json"})
    t = time.perf_counter()
    r = json.loads(urllib.request.urlopen(req, timeout=120).read())
    return r, (time.perf_counter() - t) * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:11434")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--style", choices=["json", "text"], default="json")
    ap.add_argument("models", nargs="+")
    a = ap.parse_args()
    global STYLE
    STYLE = a.style
    print(f"{'model':22s} {'tok':>4s} {'p50ms':>6s} {'p90ms':>6s} {'acc':>5s} {'acc_fright':>10s} {'conf':>5s}")
    for model in a.models:
        rng = random.Random(42)
        ask(a.base_url, model, *scenario(rng)[:2])  # warm-up / load
        lat, ok, okf, nf, conf, tok = [], 0, 0, 0, [], 0
        for _ in range(a.n):
            st, opts, good = scenario(rng)
            r, ms = ask(a.base_url, model, st, opts)
            ans = r["answers"]["move"]
            lat.append(ms); conf.append(ans["confidence"]); tok = r["usage"]["input_tokens"]
            hit = ans["choice"] == good
            ok += hit
            if st["ghosts_frightened"]:
                nf += 1; okf += hit
        lat.sort()
        print(f"{model:22s} {tok:4d} {statistics.median(lat):6.0f} {lat[int(len(lat)*0.9)]:6.0f} "
              f"{ok/a.n:5.2f} {f'{okf}/{nf}':>10s} {statistics.median(conf):5.2f}", flush=True)


if __name__ == "__main__":
    main()
