#!/usr/bin/env python3
"""Compare a teacher's labels with a reference teacher's on the same states, and project labeling cost.

Players are deterministic for a seed, so a `gen-data` run with the same player and seeds visits the same states
whatever the labeler. This script joins two such runs on (player, seed, tick), checks that the encoded states are
identical, and reports how often the two teachers chose the same option. From the token counts the new run recorded
(`label_input_tokens`, `label_output_tokens`), it projects the cost of labeling a full data set at given prices.

    python3 scripts/compare_teacher_labels.py --new data/api/train --ref data/q1/train \\
        --price deepinfra:0.15:1.875 --price alibaba-intl:0.50:3.00 --requests 54325

Stdlib only.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path


def load_rows(root: Path) -> list[dict]:
    rows = []
    for path in sorted(root.rglob("states.jsonl")):
        with path.open() as f:
            rows.extend(json.loads(line) for line in f if line.strip())
    return rows


def load_reports(root: Path) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(root.rglob("report.json"))]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (centre - half, centre + half)


def pct(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    return s[min(len(s) - 1, int(q * len(s)))]


def parse_price(spec: str) -> tuple[str, float, float]:
    name, pin, pout = spec.rsplit(":", 2)
    return name, float(pin), float(pout)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--new", type=Path, required=True, help="run folders labeled by the teacher under test")
    ap.add_argument("--ref", type=Path, help="run folders labeled by the reference teacher (same players and seeds)")
    ap.add_argument("--price", action="append", default=[], help="NAME:INPUT_USD_PER_M:OUTPUT_USD_PER_M, repeatable")
    ap.add_argument("--requests", type=int, default=54325, help="requests to project to (default: the published run)")
    ap.add_argument("--json", type=Path, help="also write the summary here")
    args = ap.parse_args()

    new_rows = load_rows(args.new)
    reports = load_reports(args.new)
    asked = [r for r in new_rows if not r.get("label_cached")]
    tin = [r["label_input_tokens"] for r in asked if r.get("label_input_tokens") is not None]
    tout = [r["label_output_tokens"] for r in asked if r.get("label_output_tokens") is not None]
    reasoning = sum(r.get("label_reasoning_tokens", 0) for r in asked)
    lat = [r["label_latency_ms"] for r in asked]
    costs = [r["label_cost_usd"] for r in asked if r.get("label_cost_usd") is not None]
    unlabeled = sum(rep.get("unlabeled", 0) for rep in reports)
    retried = sum(1 for r in asked if r.get("label_retries"))

    summary: dict = {
        "labeled": len(new_rows),
        "requests": len(asked),
        "unlabeled": unlabeled,
        "error_rate": unlabeled / max(1, len(new_rows) + unlabeled),
        "requests_retried_after_429_or_5xx": retried,
        "input_tokens_mean": statistics.fmean(tin) if tin else None,
        "input_tokens_p90": pct(tin, 0.9) if tin else None,
        "output_tokens_mean": statistics.fmean(tout) if tout else None,
        "output_tokens_p90": pct(tout, 0.9) if tout else None,
        "reasoning_tokens_total": reasoning,
        "latency_ms_p50": pct(lat, 0.5),
        "latency_ms_p90": pct(lat, 0.9),
        "reported_cost_usd": round(sum(costs), 6) if costs else None,
        "reported_cost_per_request_usd": sum(costs) / len(costs) if costs else None,
    }

    if args.ref:
        ref = {(r["player"], r["seed"], r["tick"]): r for r in load_rows(args.ref)}
        by_player: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        mismatched_state = 0
        confusion: Counter = Counter()
        for r in new_rows:
            o = ref.get((r["player"], r["seed"], r["tick"]))
            if o is None:
                continue
            if o["state"] != r["state"] or o["options"] != r["options"]:
                mismatched_state += 1
                continue
            same = int(o["label"] == r["label"])
            by_player[r["player"]][0] += same
            by_player[r["player"]][1] += 1
            if not same:
                confusion[(o["label"], r["label"])] += 1
        k = sum(v[0] for v in by_player.values())
        n = sum(v[1] for v in by_player.values())
        lo, hi = wilson(k, n)
        summary["agreement"] = {
            "matched_states": n,
            "unmatched_or_different_state": len(new_rows) - n,
            "different_state_same_key": mismatched_state,
            "same_label": k,
            "rate": k / n if n else None,
            "ci95": [round(lo, 4), round(hi, 4)],
            "by_player": {p: {"same": v[0], "n": v[1], "rate": v[0] / v[1]} for p, v in sorted(by_player.items())},
            "top_disagreements_ref_to_new": [[a, b, c] for (a, b), c in confusion.most_common(6)],
        }

    if tin and tout:
        mi, mo = statistics.fmean(tin), statistics.fmean(tout)
        proj = []
        for spec in args.price:
            name, pin, pout = parse_price(spec)
            per_request = (mi * pin + mo * pout) / 1e6
            proj.append(
                {
                    "provider": name,
                    "usd_per_m_input": pin,
                    "usd_per_m_output": pout,
                    "usd_per_1k_requests": round(per_request * 1000, 4),
                    f"usd_for_{args.requests}_requests": round(per_request * args.requests, 2),
                }
            )
        summary["projection"] = proj

    print(json.dumps(summary, indent=2))
    if args.json:
        args.json.write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
