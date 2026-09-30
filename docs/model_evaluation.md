# Model evaluation

This document records how we chose the in-game model, the fine-tuning target, and the teacher, and what the evidence does and does not support. It will grow as later stages produce in-game results.

## Summary

- **In-game default: `tev1:4b`.** It was the best off-the-shelf trade-off on this machine: about 160 ms per decision and 0.88 accuracy on the probe, faster and more accurate than `nimble`.
- **Fine-tuning target: a 0.8B model.** `tev1:0.8b` answers in about 65 ms, which is close to real time, but scores 0.40 on the probe, which is near chance. Fast and weak is the case where fine-tuning has the most to show.
- **Teacher: Qwen3.8-27B.** It scored 0.97 with thinking off and 1.00 with thinking on. Every answer was parseable and legal, and throughput was about 16 labels per second at concurrency 8.
- **State representation matters more than model size.** Compact per-direction facts beat a full ASCII board in both latency and decision quality. Quantization and prefix caching did not reduce latency.

The probe uses synthetic scenarios with one clearly correct move each. It measures latency, output reliability, and basic judgment. It does not measure how well a model plays. In-game results from the headless runner will replace it as the main evidence.

## Setup

- **Hardware:** Apple M3 Ultra for all local decision models. The teacher ran on a separate machine with an RTX 5090 (NVFP4 weights), reached over the local network.
- **Runtimes:** Ollama 0.35.0 (`/v1/systemone`) and ollaya 0.7.5 (same wire format, port 11435). The teacher was served through an OpenAI-compatible chat API.
- **Keep-alive:** Models stayed loaded (`keep_alive: -1`). Each run started with one warm-up request, which is excluded from the numbers.
- **Fresh states:** Every timed request used a newly randomized state. An exact repeat of a request returns in 16–60 ms because of caching, so a benchmark that repeats states overstates speed by 5–10x.

## Experiments

### 1. API smoke test and CORS

The support-ticket example from Ollama's documentation returned the documented answer (`bug`, p = 0.978). A request with `Origin: http://localhost:5173` returned `Access-Control-Allow-Origin` for that origin, so a page served from localhost can call Ollama directly.

### 2. Full ASCII board

We sent the full 28×31 board with Pac-Man and ghosts drawn in, using only legal moves as options, on 12 random positions.

| Model | Input tokens | Latency (cold) | Confidence |
|---|---|---|---|
| nimble | ~500 | ~430 ms | 0.00–0.09 |
| tev1:4b | ~460 | ~240 ms | 0.02–0.04 (3 positions logged) |
| tev1:0.8b | ~460 | ~80 ms | 0.01–0.03 (3 positions logged) |

Confidence near zero means the probability mass is spread almost evenly over the options. The models could not read the board as a picture.

### 3. Prefix caching

We tested the same ~530-token board with the static maze first and the changing part last, against the reverse order. Latency went from ~455 ms to ~420 ms, about an 8% saving. Latency scales with input tokens at about 0.8 ms per token for `nimble`, so the only effective lever is a smaller state.

### 4. Synthetic decision probe (`scripts/probe_decision_models.py`)

Each of 40 scenarios gives, for every legal exit, the distance to the nearest ghost and the pellets within 5 steps, plus whether ghosts are frightened. One exit is clearly best. In normal mode, the other exits have a ghost 1–3 steps away. In frightened mode, the best exit leads toward the edible ghost and the others lead to nothing. About a third of the scenarios are frightened. The instructions state the rules, including that frightened ghosts are edible. The state is compact JSON of about 130–250 tokens.

| Runtime | Model | Latency p50 | Latency p90 | Accuracy | Frightened cases |
|---|---|---|---|---|---|
| Ollama | tev1:4b (q8) | 156 ms | 188 ms | **0.88** | 13/13 |
| Ollama | tev1:4b (q4_K_M) | 161 ms | 188 ms | 0.88 | 13/13 |
| Ollama | nimble 9B (q8) | 270 ms | 299 ms | 0.68 | 12/13 |
| Ollama | nimble 9B (q4_K_M) | 286 ms | 306 ms | 0.60 | 12/13 |
| Ollama | tev1:0.8b (q8) | 67 ms | 83 ms | 0.42 | 9/13 |
| Ollama | tev1:0.8b (bf16) | 65 ms | 78 ms | 0.40 | 8/13 |
| ollaya | laya:en (421M encoder) | 20 ms | 25 ms | 0.40 | 11/13 |
| ollaya | kev:0.8b | 729 ms | 970 ms | 0.40 | 9/13 |

Random choice over 2–4 legal options scores about 0.36 on this set. Observations:

- `tev1:4b` is both faster and more accurate than `nimble`. Its advantage is in normal-mode scenarios, where it has to trade pellets against a nearby ghost.
- q4 quantization did not speed anything up. A decision outputs a single token, so the time goes into reading the input, which is limited by compute, not memory bandwidth.
- The 0.8B-class models and `laya:en` are near chance. `laya:en` does well when ghosts are frightened (move toward the ghost) and poorly otherwise.
- `kev:0.8b` ran on ollaya's CPU path on this Mac, which makes it too slow for real-time play here, whatever its accuracy.

### 5. Plain English instead of JSON

We rendered the same facts as sentences ("Going left: a ghost 2 steps away, 3 pellets in the next 5 steps."). It did not help. `laya:en` went from 0.40 to 0.47, and `tev1:4b` dropped from 0.88 to 0.78. We keep compact JSON.

### 6. Teacher: Qwen3.8-27B (`scripts/probe_teacher.py`)

Same 40 scenarios, same rules, same JSON state. The prompt asks for `{"move": ..., "reason": ...}`. Temperature 0.

| Mode | Latency p50 | Latency p90 | Parseable | Legal | Accuracy | Output tokens |
|---|---|---|---|---|---|---|
| Thinking off | 270 ms | 309 ms | 40/40 | 40/40 | 0.97 | ~36 |
| Thinking on (concurrency 4) | 1,135 ms | 1,553 ms | 40/40 | 40/40 | 1.00 | ~190 |
| Thinking off, concurrency 8 (80 requests) | 405 ms | 731 ms | 80/80 | 80/80 | 0.97 | ~36 |

Throughput at concurrency 8 was 16 labels per second. The server does not support `response_format: json_schema`, because it has no constrained decoding, but a prompt that asks for JSON was parseable in all 120 calls. The two misses with thinking off both went for more pellets at the cost of ghost distance, which is the same kind of trade-off where the small models fail.

### 7. Prior art: Ollama's Pac-Man replay

Ollama's launch post embeds a recorded game as JSON: `nimble:9b-int4`, a 19×21 maze, three ghosts, one decision per tile step (turn-based), 187 moves, 91 ms mean latency on an M5 Max, median confidence 0.58, ending with Pac-Man caught. It confirms that the legal-moves-only option set works. It ships no engine, prompt, or frontend to reuse.

## Conclusions

1. Use `tev1:4b` for the real-time demo. At about 160 ms, a decision is shorter than the typical 400 ms to 1.2 s Pac-Man needs to reach the next junction at arcade speed.
2. Use a 0.8B model as the fine-tuning target and `tev1:0.8b` as its off-the-shelf baseline.
3. Use Qwen3.8-27B as the teacher, with the engine's lookahead oracle as a cross-check. The probe is too easy to tell how the teacher does on ambiguous real positions.
4. Invest in the encoder before the model. Keep states compact and factual, and measure every encoder change in tokens as well as accuracy.

## Pending

- ollaya models still downloading at the time of writing: `kev` (4B), `kev:9b`, `decider` (0.8B, 2B, 4B), and `winnow:e4b`. Their probe rows will be added here.
- In-game results from the headless runner (Stage 1) replace the probe as the main evidence for the model choice.
