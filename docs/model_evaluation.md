# Model evaluation

This document records how we chose the in-game model, the fine-tuning target, and the teacher, and what the evidence does and does not support. It will grow as later stages produce in-game results.

## Summary

- **In-game default: `tev1:4b`.** It was the best off-the-shelf trade-off on this machine: about 160 ms per decision and 0.88 accuracy on the probe, faster and more accurate than `nimble`.
- **Fine-tuning target: a 0.8B model.** `tev1:0.8b` answers in about 65 ms, which is close to real time, but scores 0.40 on the probe, which is near chance. Fast and weak is the case where fine-tuning has the most to show.
- **Labels: a search oracle, not an LLM.** An agent that rolls the game engine forward 5 seconds for each option survived every 5-minute evaluation game and averaged 1,109 pellets, at 5 ms of local CPU per decision. Qwen3.8-27B looked strong on the synthetic probe (0.97), but as a player it lost all three lives within 39 to 93 seconds in 5-minute games.
- **Fine-tuning works.** A 0.8B model fine-tuned on 31,585 oracle-labeled states (`pacman-0.8b`) averaged 456 pellets and 94 s per game at 63 ms per decision. That is 6.6 times off-the-shelf `tev1:0.8b` at the same speed, 3 times `tev1:4b`, and more than twice the scripted greedy rule. The oracle it learned from still averages 1,109.
- **Hosted Jev sits between `tev1:4b` and the greedy rule.** Through TypeSafe's API, Jev 1.13.0 averaged 178 pellets and 52 s per game at 115 ms per decision, and agreed with the oracle on 39.9% of held-out states. The fine-tuned 0.8B model beats it on both.
- **State representation matters more than model size.** Compact per-direction facts beat a full ASCII board in both latency and decision quality. Quantization and prefix caching did not reduce latency.

The probe uses synthetic scenarios with one clearly correct move each. It measures latency, output reliability, and basic judgment. It does not measure how well a model plays. Experiments 8 to 10 measure play in the game itself and carry more weight.

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
| ollaya | winnow:e4b (GGUF, Metal) | 154 ms | 170 ms | 0.78 | 9/13 |
| ollaya | decider:0.8b | 330 ms | 447 ms | 0.35 | 5/13 |
| ollaya | decider (2B) | 681 ms | 955 ms | 0.60 | 8/13 |
| ollaya | decider:4b | 2,186 ms | 2,724 ms | 0.65 | 12/13 |
| ollaya | kev (4B) | 2,052 ms | 2,672 ms | 0.50 | 13/13 |
| ollaya | kev:9b | 3,940 ms | 4,721 ms | 0.57 | 13/13 |

Random choice over 2–4 legal options scores about 0.36 on this set. Observations:

- `tev1:4b` is both faster and more accurate than `nimble`. Its advantage is in normal-mode scenarios, where it has to trade pellets against a nearby ghost.
- q4 quantization did not speed anything up. A decision outputs a single token, so the time goes into reading the input, which is limited by compute, not memory bandwidth.
- The 0.8B-class models and `laya:en` are near chance. `laya:en` does well when ghosts are frightened (move toward the ghost) and poorly otherwise.
- On this Mac, ollaya runs `kev` and `decider` on the CPU, which puts them at 0.3–4 s per decision, too slow for real-time play here whatever their accuracy. `winnow:e4b` ships as GGUF and runs on llama.cpp with Metal. It is the one ollaya model in `tev1:4b`'s latency range, at 0.78 against 0.88.

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

### 8. In-game results (Stage 1, headless)

`scripts/run_headless.ts`, 1x speed, realtime clock (model latency turns into game time), features encoder, junction lookahead, seeds starting at 100. Level 1 has 244 pellets.

| Policy | Games | Mean pellets | Mean score | Mean survival | Decision p50 | Stale answers |
|---|---|---|---|---|---|---|
| random | 10 | 30.3 | 311 | 37 s | 0 ms | 0% |
| greedy (scripted) | 10 | 200.7 | 2,623 | 63 s | 0 ms | 0% |
| tev1:4b | 3 | 168.7 | 1,833 | 57 s | 207 ms | 10% |

tev1:4b plays far above random and below the scripted baseline, which reads the same facts with hand-written rules. This is the gap the teacher data and fine-tuning stages are meant to close for a 0.8B model. Three games are enough to show that play works, not to rank models. Stage 4 will use more seeds.

### 9. Qwen3.8-27B as a player (`gen_teacher_data.ts`, lockstep)

The teacher gets the same encoded state as the student and answers `{"move", "reason"}`. Lockstep means the game waits for each answer, so latency does not count against it.

30 seconds per game, seeds 1001-1003:

| Qwen sees | Thinking | Pellets (3 games) | Mean pellets | Deaths (total) | Unparseable |
|---|---|---|---|---|---|
| Features | off | 155, 182, 170 | 169 | 2 | 0 |
| Features | on | 165, 181, 182 | 176 | 0 | 13 of ~500 |
| Full ASCII map (`ascii-full`) | off | 105, 107, 85 | 99 | 6 | 0 |

The full-map run with thinking on was stopped after 48 minutes, too slow to matter.

**Qwen misreads the ASCII map.** Its reasons mention dead ends, which this maze does not have, and a tunnel on row 29, when the tunnel is on row 14. It also headed for a distant power pellet when one sat on the same row. Reading vertical relationships from a grid serialized as text is unreliable: tiles above and below each other sit dozens of tokens apart.

**30 seconds hides the difference between players.** In 5-minute lockstep games on seeds 100-102, Qwen (features, thinking off) ate 361, 361, and 180 pellets. It lost all three lives within 74, 93, and 39 seconds.

### 10. Search baselines

**Survey.** A literature survey of simple, non-RL Pac-Man agents found a consistent lesson. The strongest simple agents rely on two things: an accurate model of ghost behavior, and a "safe path" test (Pac-Man reaches the next junction before any dangerous ghost). Supporting evidence:

- A model-based Ms. Pac-Man agent predicted ghost moves with 94.6% accuracy and averaged 38,172 points on the real game, above the competition record of the time. Sources: [Cornell news](https://news.cornell.edu/stories/2017/01/engineers-eat-away-ms-pac-man-score-artificial-player), survey in IEEE ToG 2018 ([ResearchGate](https://www.researchgate.net/publication/321821472)).
- The MCTS agent that won CIG'12 averaged 82,689 against the framework's baseline ghosts. Its ablations show that most of its strength comes from rule-based, safety-aware playouts: random playouts cut the score by 60-83%. Source: [Pepels, Winands and Lanctot](https://dke.maastrichtuniversity.nl/m.winands/documents/CIG2012_paper_106.pdf).
- Rule-based agents with A* over ghost-weighted path costs (ICE Pambush) won CEC'09 with a 13,059 average. Source: [Matsumoto et al.](https://www.ice.ci.ritsumei.ac.jp/~ruck/PAP/jsst09-matsumoto.pdf).
- The UC Berkeley CS188 Pac-Man agents are teaching templates, not strength benchmarks. The project only asks for more than half wins against one random ghost on a small layout. Source: [CS188 project 2](https://inst.eecs.berkeley.edu/~cs188/fa24/projects/proj2/).
- Our `greedy` rule matches the framework's starter agent: flee, chase edible ghosts, else go to the nearest pellet.

Published scores come from real Ms. Pac-Man or the Java competition framework, so they are not comparable with ours. Only the ranking of techniques transfers. Our engine is easier than both: ghost moves are deterministic except in frightened mode, so an agent can simulate the future exactly.

**Our oracle** (`src/agent/oracle.ts`). For each option, copy the game, play the option, and run the real engine forward for a fixed horizon, with greedy play at later junctions. The value is the score gained, minus a large penalty for a death that shrinks the later the death happens, minus the distance to the nearest pellet at the end.

Evaluation seeds 100-109, realtime clock, 1x speed, 5-minute cap:

| Horizon | Mean pellets | Mean score | Games reaching the cap | Decision p50 |
|---|---|---|---|---|
| 3 s | 1,009 | 16,546 | 10 of 10 | 3 ms |
| **5 s** | **1,109** | **19,789** | 10 of 10 | 5 ms |
| 8 s | 1,011 | 15,858 | 10 of 10 | 7 ms |

A level has 244 pellets, so 1,109 is about four and a half levels. Horizons past 5 seconds do not help, probably because the greedy playout policy gets less reliable further out. That explanation is not tested.

5-minute lockstep head-to-head, seeds 100-102:

| Player | Pellets | Result |
|---|---|---|
| oracle-3s | 971, 1,178, 1,211 | 1 death per game, all reached the cap |
| Qwen3.8-27B (features, thinking off) | 361, 361, 180 | all lives lost at 74, 93, 39 s |

### 11. Fine-tuned 0.8B (`pacman-0.8b`)

`Qwen/Qwen3.5-0.8B` fine-tuned with LoRA on oracle-labeled states in the exact prompt format Ollama uses (see RFC, Stage 3). It is served as Q8_0 GGUF through Ollama like the stock models.

**Agreement with the oracle**, 1,000 validation states from seeds 900-999 (never used for training), scored through Ollama. "Decisive" means the oracle's soft target puts more than 0.9 on one option.

| Model | Agreement | On decisive states | Cross-entropy | Latency p50 |
|---|---|---|---|---|
| Qwen3.5-0.8B before training (training-side eval, 2,000 states) | 28.8% | 30.6% | 1.43 | - |
| `tev1:0.8b` | 31.8% | 29.5% | 1.22 | 73 ms |
| `tev1:4b` | 38.4% | 40.7% | 1.29 | 205 ms |
| Jev 1.13.0 (hosted) | 39.9% | 40.9% | 1.41 | 237 ms |
| **`pacman-0.8b`** | **53.9%** | **57.5%** | **1.05** | **59 ms** |

Validation agreement rose from 29% to 53% within 1,200 of 1,975 steps and then flattened. The plateau is likely the information gap: many oracle decisions depend on facts the features encoder does not include.

**In-game**, evaluation seeds 100-109, realtime clock, 1x speed, 5-minute cap:

| Player | Mean pellets | Mean score | Mean survival | Levels cleared | Decision p50 | Stale |
|---|---|---|---|---|---|---|
| random | 30 | 311 | 37 s | 0 of 10 | 0 ms | 0% |
| `tev1:0.8b` | 69 | 716 | 35 s | 0 of 10 | 62 ms | 4% |
| `tev1:4b` | 148 | 1,660 | 55 s | 0 of 10 | 202 ms | 11% |
| Jev 1.13.0 (hosted) | 178 | 2,482 | 52 s | 0 of 10 | 115 ms | 3% |
| greedy (scripted) | 201 | 2,623 | 63 s | 0 of 10 | 0 ms | 0% |
| **`pacman-0.8b`** | **456** | **7,467** | **94 s** | **8 of 10** | **63 ms** | **3%** |
| `oracle-5s` (teacher) | 1,109 | 19,789 | 300 s (cap) | 10 of 10 | 5 ms | 0% |

Per game, `pacman-0.8b` ate 239, 400, 417, 649, 482, 683, 587, 420, 240, and 446 pellets. The best game reached level 3.

**Parity between training and serving.** On 50 validation states, the Q8_0 model in Ollama and the bf16 weights in PyTorch agree to within 0.027 in probability (median 0.010), and pick the same option on 48 of 50. The two that differ are near-ties.

**Recordings**, all seed 100 at 1x for 30 s:

| Clip | Model | Score | Pellets | Lives lost |
|---|---|---|---|---|
| `docs/media/demo_tev1_08b.mp4` | `tev1:0.8b` | 710 | 71 | 2 |
| `docs/media/demo_tev1_4b.mp4` | `tev1:4b` | 1,250 | 101 | 1 |
| `docs/media/demo_jev.mp4` | Jev 1.13.0 (hosted) | 900 | 86 | 1 |
| `docs/media/demo_pacman_08b.mp4` | `pacman-0.8b` | 2,660 | 170 | 0 |

In the `tev1:0.8b` clip, the option probabilities stay close to uniform (confidence 0.00-0.04).

### 12. Jev, TypeSafe's hosted decision model

Jev is served at `https://api.typesafe.ai/v1/systemone` with the same request and response format as Ollama's endpoint, so the game calls it unchanged. Only the base URL, the model name (`jev-latest`, which answered as `jev-1.13.0`), and a bearer key differ. The API rejects Ollama's `keep_alive` field, so the client leaves it out for hosted endpoints. Run it with `TYPESAFE_API_KEY` set: `npm run headless -- --policy jev`, or `VITE_DECISION_BASE_URL=https://api.typesafe.ai npm run dev` and `?model=jev-latest` in the browser. The dev server adds the key to proxied requests, so it never reaches the page.

**Latency.** Over the internet from the test machine, p50 was 106-123 ms per game in the headless run. The agreement run, which sent requests while the games were also running, measured p50 237 ms and p90 298 ms. That is comparable to `tev1:4b` running locally, which pays for its latency with local compute instead of network.

**Play.** Per game, Jev ate 208, 173, 165, 198, 125, 203, 103, 231, 193, and 179 pellets, and lost all three lives in every game. It plays better than `tev1:4b` (148) and slightly worse than the greedy rule (201).

**Tokens.** The same state costs about 605 input tokens on Jev against about 380 on the local models. A trivial request costs 325, so the hosted service adds roughly 300 tokens of its own per call. The 10 games used 2.05 million input tokens over 3,368 decisions.

## Conclusions

1. Use `tev1:4b` for the real-time demo. At about 160 ms, a decision is shorter than the typical 400 ms to 1.2 s Pac-Man needs to reach the next junction at arcade speed.
2. Use a 0.8B model as the fine-tuning target and `tev1:0.8b` as its off-the-shelf baseline. The fine-tuned `pacman-0.8b` beats every off-the-shelf model and the scripted rule in play, at `tev1:0.8b`'s latency.
3. Label training data with `oracle-5s`. It is far stronger than any model we tried and costs milliseconds of local CPU per label. It also scores every option, so labels can be soft: ties become visible instead of being broken arbitrarily. Keep Qwen3.8-27B only as a comparison point.
4. Mind the information gap. The oracle sees the full game state, and the student sees only its encoding. Decisions that depend on facts the encoding drops cannot be learned from it. Measure the student against the oracle on held-out states to see how large this gap is.
5. Invest in the encoder before the model. Keep states compact and factual, and measure every encoder change in tokens as well as accuracy.

## Possible follow-ups (not planned)

The project stops here: the demo sequence (off-the-shelf 0.8B, off-the-shelf 4B, fine-tuned 0.8B) is complete. If it resumes, the gap to the oracle is the obvious target:

- Enrich the encoder: ghost identities, the second-nearest ghost, and longer-range pellet counts.
- Run a DAgger round with states from `pacman-0.8b`'s own games.
- Try a full-map student: fine-tune on `ascii-full` with oracle labels and compare.
