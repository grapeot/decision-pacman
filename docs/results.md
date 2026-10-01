# Results

This document reports pellets eaten per game and decision latency across players. The full lab notebook is [model_evaluation.md](model_evaluation.md), whose section numbers are cited below.

## How to read these numbers

All ladder benchmarks use the `features` input without lookahead. Evaluations run across 10 games on seeds 100-109 at 1x speed with a 5-minute cap per game under the realtime clock. A single level contains 244 pellets. "Pellets" indicates the mean number of pellets eaten per game. Latency is the median decision time (p50).

The runtime supports two clocks:
- `realtime` (default): measured decision latency translates into game ticks that pass before an action applies. Late decisions arrive after Pac-Man has already passed a junction ("stale").
- `lockstep`: the simulation waits for every answer, measuring decision quality independent of latency.

Lockstep results are not comparable with realtime results. The same holds for browser play with the Wait switch on, which holds the game at a junction until the model answers.

The lookahead rule ([AGENTS.md](../AGENTS.md); conclusion 10 of [model_evaluation.md](model_evaluation.md)): lookahead (engine rollouts) may label training data but never enters a player's input in a comparison. Players are compared on the same current-state input (`features`). An exact rollout of a deterministic engine is close to seeing the future (only frightened ghosts turn at random). Results with lookahead in the input are records, not ladder rows.

Hardware: local models ran on an Apple M3 Ultra with Ollama 0.35.0. Jev ran over the internet against TypeSafe's hosted API. The Qwen3.8-27B teacher and all fine-tuning ran on one NVIDIA RTX 5090.

## The ladder

All rows: `features` input, no lookahead, 10 games on seeds 100-109, 1x speed, 5-minute cap, realtime clock.

| Player | Mean pellets | Mean survival | Decision p50 | Source |
|---|---|---|---|---|
| random | 30 | 37 s | 0 ms | section 11 |
| `tev1:0.8b` | 69 | 35 s | 62 ms | section 11 |
| `tev1:4b` | 148 | 55 s | 202 ms | section 11 |
| Jev 1.13.0 (hosted) | 178 | 52 s | 115 ms | sections 11, 12 |
| `llm:qwen3.5:4b` | 197 | 47 s | 267 ms | section 14 |
| greedy (scripted) | 201 | 63 s | 0 ms | section 11 |
| `llm:gemma4:e4b` | 230 | 55 s | 253 ms | section 14 |
| `llm:phi4-mini` | 232 | 55 s | 187 ms | section 14 |
| `pacman-0.8b` | 386 | 81 s | 59-63 ms | sections 11, 18 |
| `pacman-0.8b-qwen` | 425 | 86 s | 55 ms | section 18 |

The fine-tuned rows report the mean of three 10-game runs: `pacman-0.8b` scored 456, 300, and 400 pellets; `pacman-0.8b-qwen` scored 441, 467, and 366 pellets. Model weights and training datasets are not published in the repository; to train them, follow [guides/distill.md](guides/distill.md).

For commands to run ladder evaluations, see [guides/evaluate.md](guides/evaluate.md).

## Lockstep results

Under the lockstep clock, the simulation pauses until the model returns an answer. These results isolate move quality from latency and do not belong on the ladder.

| Player | Seeds | Mean pellets | Decision p50 | Source |
|---|---|---|---|---|
| Qwen3.8-27B, features, thinking off | 100-102 | 301 | - | sections 10, 13 |
| Qwen3.8-27B, features, thinking on | 100-102 | 531 | 2.4 s (p90 8.6 s) | section 13 |
| `pacman-0.8b-qwen` | 100-109 | 417 | - | section 18 |

The Qwen evaluations covered 3 games on seeds 100-102 (individual scores: 361, 361, 180 with thinking off). Qwen ran on an RTX 5090 (NVFP4) over the local network.

## Records with lookahead in the input

In these runs the player's input included engine rollouts (lookahead), or the player computed them at decision time (`oracle-5s`). They are kept as records and as evidence of label quality. They are not compared with the ladder.

| Player | Clock | Mean pellets | Source |
|---|---|---|---|
| `oracle-5s` | realtime | 1,109 | section 10 |
| Qwen3.8-27B (`teacher-peek5s`) | lockstep | 1,396 | section 13 |
| Student B (`pacman-0.8b-qwen-peek`) | lockstep | 1,387 | section 18 |
| Student B (`pacman-0.8b-qwen-peek`) | realtime | 765 | section 18 |
| Jev (`features-peek5s`) | realtime | 567 | section 16 |
| `tev1:4b` (`features-peek5s`) | realtime | 212 | section 16 |

`oracle-5s` rolls the engine forward 5 seconds per option in about 5 ms CPU time; rollout variants `oracle-3s` and `oracle-8s` scored 1,009 and 1,011 pellets, respectively.

For label quality, the Qwen teacher with 5-second lookahead (`teacher-peek5s`) selected the oracle's top choice on 90.7% of training states, 92.0% of validation states, and 88.3% on section 11's 1,000 validation states (88-92% overall).

## Agreement with the oracle

This offline test measures how often a model selects the option preferred by `oracle-5s` across 1,000 held-out validation states. It serves as a secondary check to in-game performance.

| Player | Agreement | Source |
|---|---|---|
| `tev1:0.8b` | 31.8% | section 11 |
| `tev1:4b` | 38.4% | section 11 |
| Jev 1.13.0 | 39.9% | section 11 |
| `llm:gemma4:e4b` | 43.6% | section 14 |
| `llm:phi4-mini` | 44.0% | section 14 |
| `llm:qwen3.5:4b` | 45.4% | section 14 |
| `pacman-0.8b` | 53.9% | sections 11, 18 |
| `pacman-0.8b-qwen` | 55.3% | section 18 |

## Generality outside Pac-Man

Accuracy on 194 public JevBench decision items (non-Pac-Man tasks including support routing, intent detection, policy checks, and severity classification; items exceeding 4,000 characters skipped).

| Model | Accuracy | Source |
|---|---|---|
| `tev1:4b` | 0.81 | section 15 |
| `tev1:0.8b` | 0.66 | section 15 |
| `pacman-0.8b-qwen` | 0.53 | section 18 |
| `pacman-0.8b` | 0.49 | section 18 |
| Chance baseline | 0.32 | section 15 |
| Jev 1.13.0 (JevBench's published run) | 0.89 | section 15 |

Jev's 0.89 is computed from JevBench's published per-item results on the same 194 items, not measured here. The evaluation script is `scripts/eval_general_decisions.py`.

## Where decision time goes

Latency medians measured on 20 game prompts of approximately 383 tokens (section 17).

| Metric | M3 Ultra, Ollama | RTX 5090, llama-server | RTX 5090, llama-bench |
|---|---|---|---|
| `tev1:4b` prefill | 178 ms | 50 ms | 25 ms |
| `tev1:4b` each further output token | 12.0 ms | 4.8 ms | 4.5 ms |
| `tev1:4b` end to end, 1 output token | 197 ms | 124 ms | - |
| `pacman-0.8b` prefill | 41 ms | 17 ms | 11 ms |
| `pacman-0.8b` each further output token | 5.5 ms | 2.1 ms | 2.0 ms |
| `pacman-0.8b` end to end, 1 output token | 54 ms | 47 ms | - |

A decision is prefill-bound. A decision model reads its answer directly from option logits in one forward pass, whereas a chat model generating a ~6-token JSON response spends an additional ~72 ms on the M3 Ultra and ~27 ms on the RTX 5090 in decoding steps.

## Variance and caveats

Realtime scores for the same model varied by up to about 150 pellets across 10-game runs. For example, `pacman-0.8b` recorded runs of 456, 300, and 400 pellets. The 2026-10-01 runs shared the host Mac with heavy unrelated background workload (load average 15-36); `pacman-0.8b` and `pacman-0.8b-qwen` runs were alternated to face matching load conditions. Compare realtime scores within the same session rather than across days.

Single games show wide variance. Jev's 10 evaluation games ranged from 103 to 231 pellets.

The realtime clock charges latency as game time, so it handicaps slower players, such as the chat models at 187-267 ms per decision.

Repeated identical requests hit inference caching and return 5-10x faster, so reported latency numbers require fresh game states.

The clips below are single 30-second games. They illustrate play; they do not measure it.

## Clips

All clips record 30 seconds of gameplay on seed 100 at 1x speed.

| File | Player | Score | Pellets | Lives lost |
|---|---|---|---|---|
| [`media/demo_tev1_08b.mp4`](media/demo_tev1_08b.mp4) | `tev1:0.8b` | 710 | 71 | 2 |
| [`media/demo_tev1_4b.mp4`](media/demo_tev1_4b.mp4) | `tev1:4b` | 1,250 | 101 | 1 |
| [`media/demo_jev.mp4`](media/demo_jev.mp4) | Jev 1.13.0 (hosted) | 900 | 86 | 1 |
| [`media/demo_qwen35_4b.mp4`](media/demo_qwen35_4b.mp4) | `llm:qwen3.5:4b` | 1,370 | 109 | 0 |
| [`media/demo_pacman_08b.mp4`](media/demo_pacman_08b.mp4) | `pacman-0.8b` | 2,660 | 170 | 0 |
