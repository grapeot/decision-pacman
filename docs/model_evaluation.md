# Model evaluation

This document records how we chose the in-game model, the fine-tuning target, and the teacher, and what the evidence does and does not support. It will grow as later stages produce in-game results.

## Summary

- **In-game default: `tev1:4b`.** It was the best off-the-shelf trade-off on this machine: about 160 ms per decision and 0.88 accuracy on the probe, faster and more accurate than `nimble`.
- **Fine-tuning target: a 0.8B model.** `tev1:0.8b` answers in about 65 ms, which is close to real time, but scores 0.40 on the probe, which is near chance. Fast and weak is the case where fine-tuning has the most to show.
- **Labels: a search oracle, not an LLM.** An agent that rolls the game engine forward 5 seconds for each option survived every 5-minute evaluation game and averaged 1,109 pellets, at 5 ms of local CPU per decision. Qwen3.8-27B looked strong on the synthetic probe (0.97), but as a player it lost all three lives within 39 to 93 seconds in 5-minute games.
- **Fine-tuning works.** A 0.8B model fine-tuned on 31,585 oracle-labeled states (`pacman-0.8b`) averaged 456 pellets and 94 s per game at 63 ms per decision. That is 6.6 times off-the-shelf `tev1:0.8b` at the same speed, 3 times `tev1:4b`, and more than twice the scripted greedy rule. The oracle it learned from still averages 1,109.
- **The specialization is not free.** On 194 public JevBench decisions that are not Pac-Man (support routing, intent, policy checks, severity), `pacman-0.8b` scored 0.49 against 0.66 for `tev1:0.8b` and 0.81 for `tev1:4b`, with chance at 0.32. It loses most on the easy and standard tiers, where `tev1:0.8b` is competent, and its probabilities there stay close to uniform. On the hard tier both 0.8B models are near chance.
- **Hosted Jev sits between `tev1:4b` and the greedy rule.** Through TypeSafe's API, Jev 1.13.0 averaged 178 pellets and 52 s per game at 115 ms per decision, and agreed with the oracle on 39.9% of held-out states. The fine-tuned 0.8B model beats it on both.
- **An ordinary small chat model plays as well as Jev.** Given the same facts, a JSON schema restricted to the legal options, and thinking off, stock `phi4-mini` averaged 232 pellets at 187 ms per decision, `gemma4:e4b` 230, and `qwen3.5:4b` 197, against Jev's 178, all in real time with no failed answers. All three also agreed with the oracle more often than Jev (44-45% against 39.9%).
- **A general LLM with lookahead beats the search oracle.** Qwen3.8-27B, shown one 5-second simulated future per option, averaged 1,396 pellets and reached level 6 in all 10 games, against 1,109 for `oracle-5s`, which scores the same rollouts with a hand-written formula. Without the lookahead, thinking took Qwen from 301 to 531 pellets, but it still lost all its lives within about two minutes.
- **Closed models can use lookahead too, with very different gains.** Given the same 5-second rollout outcomes per option, hosted Jev went from 178 to 567 pellets in real time, beating the fine-tuned `pacman-0.8b` on plain features (456), while `tev1:4b` only went from 148 to 212. A fixed formula over the same rollouts (`oracle-5s`, 1,109) still beats Jev; Qwen3.8-27B (1,396) beats the formula.
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

### 13. Qwen3.8-27B with thinking, and with lookahead

Two ways to give the general model more to work with, both in lockstep (the game waits for each answer), 1x speed, 5-minute cap, features encoder.

- **Thinking** (`teacher-think`): the same prompt as section 9, with thinking on.
- **Lookahead** (`teacher-peek5s`, thinking off): for each option, the engine plays that option and then greedy moves for 5 s, exactly as the oracle's rollout does. The prompt adds the outcome per option as facts: seconds until death (or none), pellets, power pellets, ghosts eaten, and points. The model makes the choice.

| Player | Seeds | Mean pellets | Mean score | Mean survival | Decision p50 | Failed answers |
|---|---|---|---|---|---|---|
| Qwen, thinking off (section 10) | 100-102 | 301 | - | 69 s | - | - |
| Qwen, thinking on | 100-102 | 531 | 7,217 | 114 s | 2.4 s (p90 8.6 s) | 34 of 2,167 |
| `oracle-5s` (realtime) | 100-109 | 1,109 | 19,789 | 300 s (cap) | 5 ms | - |
| **Qwen, lookahead** | 100-109 | **1,396** | **22,054** | 299 s | 309 ms | 11 of 15,668 |

Per game, Qwen with lookahead ate 1,457, 1,372, 1,364, 1,321, 1,427, 1,412, 1,363, 1,391, 1,420, and 1,431 pellets, and every game reached level 6. Nine games reached the cap; seed 102 lost its last life at 292 s. Failed answers fall back to the previous intent.

Both players see the same rollouts. The oracle reduces each to one number (points, minus a large penalty for dying, minus the distance to the nearest pellet) and takes the highest. The model reads the outcomes and chooses. That it does better says the hand-written value function is the oracle's weak part. Which trade-offs the model makes differently has not been analyzed.

Thinking alone helps (301 to 531 pellets) but costs about 9 times the latency, and the model still misjudges danger from the encoded facts alone.

### 14. Plain small chat models with structured output

Does Jev, or Tev1, do something an ordinary small model cannot? `llm:<model>` (`src/agent/llm.ts`) asks a general chat model through Ollama's `/api/chat` with the facts Jev sees: the features state, the same instructions, and the same options, in the teacher's prompt from section 9 without the reason field. A JSON schema restricts `move` to the legal options, so every answer is legal. Thinking is off, temperature is 0, and the model stays loaded. Option probabilities come from the token logprobs at the position where the move is written, renormalized over the options. The clock is realtime, as it was for Jev.

The models are stock Ollama library builds (Q4_K_M), pulled for this test:

- `qwen3.5:4b` (4.7B): the base model that Tev1's 4B LoRA is trained on.
- `gemma4:e4b` (7.5B stored, about 4B used per token).
- `phi4-mini` (3.8B).

**In-game**, evaluation seeds 100-109, realtime clock, 1x speed, 5-minute cap, features encoder:

| Player | Mean pellets | Mean score | Mean survival | Levels cleared | Decision p50 | Stale | Failed answers |
|---|---|---|---|---|---|---|---|
| random | 30 | 311 | 37 s | 0 of 10 | 0 ms | 0% | - |
| `tev1:4b` | 148 | 1,660 | 55 s | 0 of 10 | 202 ms | 11% | - |
| Jev 1.13.0 (hosted) | 178 | 2,482 | 52 s | 0 of 10 | 115 ms | 3% | - |
| `llm:qwen3.5:4b` | 197 | 2,553 | 47 s | 0 of 10 | 267 ms | 9% | 0 of 1,553 |
| greedy (scripted) | 201 | 2,623 | 63 s | 0 of 10 | 0 ms | 0% | - |
| `llm:gemma4:e4b` | 230 | 3,276 | 55 s | 1 of 10 | 253 ms | 4% | 0 of 1,929 |
| `llm:phi4-mini` | 232 | 3,142 | 55 s | 2 of 10 | 187 ms | 4% | 0 of 2,468 |
| `pacman-0.8b` (fine-tuned) | 456 | 7,467 | 94 s | 8 of 10 | 63 ms | 3% | - |

Per game, `qwen3.5:4b` ate 213, 242, 185, 152, 239, 219, 122, 215, 217, and 169 pellets; `gemma4:e4b` 220, 242, 304, 196, 211, 238, 232, 232, 195, and 230; `phi4-mini` 170, 235, 240, 226, 215, 269, 221, 144, 371, and 231. Every game ended with all three lives lost. Paired by seed against Jev, the mean difference is +20 pellets for `qwen3.5:4b` (standard error 14), +52 for `gemma4:e4b` (17), and +54 for `phi4-mini` (24).

**Agreement with the oracle**, the same 1,000 validation states as section 11, asked the same way as in the game (`training/score_ollama.py --chat`):

| Model | Agreement | On decisive states | Cross-entropy | Latency p50 |
|---|---|---|---|---|
| `tev1:4b` | 38.4% | 40.7% | 1.29 | 205 ms |
| Jev 1.13.0 (hosted) | 39.9% | 40.9% | 1.41 | 237 ms |
| `llm:gemma4:e4b` | 43.6% | 43.6% | - | 245 ms |
| `llm:phi4-mini` | 44.0% | 43.6% | 1.72 | 148 ms |
| `llm:qwen3.5:4b` | 45.4% | 46.4% | 1.67 | 237 ms |
| `pacman-0.8b` | 53.9% | 57.5% | 1.05 | 59 ms |

Observations:

- All three plain models play at least as well as Jev in real time, and none produced a failed answer. `gemma4:e4b` and `phi4-mini` also beat the greedy rule. `qwen3.5:4b`'s lead over Jev is within the noise of 10 games; the other two lead by about two to three standard errors. None comes near the fine-tuned 0.8B model.
- Plain `qwen3.5:4b` beats `tev1:4b`, which is the same base with a decision LoRA, both in play (197 against 148 pellets) and in agreement (45.4% against 38.4%). The two differ in more than the LoRA: chat with a generated JSON answer against System One's scoring of option letters, and Q4_K_M against q8 weights. Which of these accounts for the gap was not tested.
- The plain models are slower than Jev, not faster. A chat answer generates about six tokens (`{"move": "left"}`) after reading about 300 input tokens, which takes 190-270 ms locally against Jev's 115 ms over the network. The realtime clock therefore handicaps them. `qwen3.5:4b` is the slowest of the three and has the highest stale rate (9%).
- Probabilities are available from logprobs for `qwen3.5:4b` and `phi4-mini` (median confidence 0.76 and 0.77). Ollama 0.35.0 returns logprobs only for the first generated token of `gemma4:e4b`, so it has none. The chat probabilities are more overconfident than the System One ones: cross-entropy against the oracle's soft target is 1.67-1.72, against 1.29-1.41. They are fine for the probability bars, not as calibrated scores.
- What this shows: on this task, given the same facts, Jev has no advantage that a stock 4B chat model with constrained JSON output lacks. It does not test Jev's other selling points, such as several questions per request or calibration on other tasks. It also says something about the task: every player that reads only the features encoder, scripted, decision model, or chat model, lands between 148 and 232 pellets, while the fine-tuned 0.8B model reaches 456 from the same facts.

### 15. What specialization cost

`pacman-0.8b` plays far better than `tev1:0.8b`. This experiment measures the other side: how much worse it is at decisions that are not Pac-Man.

**Evaluation set.** The public items of [JevBench](https://github.com/fstandhartinger/jevbench) (MIT), a third-party benchmark for decision models in the same `/v1/systemone` format, at commit `bb05a33`. They are labeled short decisions about refund policies, support messages, tool requests, incident severity, and similar text, posed as yes/no (`noul`), pick-one (`choice`), and ordinal (`score`) questions with 2-6 options. JevBench groups them into three tiers: easy (48 items), standard (72), and hard (111, written to contain traps and multi-step reasoning). The models in Ollama were loaded with a 2,050-token context, so items longer than 4,000 characters were skipped: 37 hard items, leaving 194. `scripts/eval_general_decisions.py` downloads the files, checks their hashes, and sends each item once with JevBench's own question. Each item is scored by argmax, as JevBench does, and by the probability the model gave the correct option. Ollama turns all three question types into the same lettered-choice prompt that `pacman-0.8b` was trained on, so the format itself is not new to it.

**Results.** Accuracy, with the probability on the correct option in parentheses. Chance is the mean of 1 / options per item. Latency is p50 on this set (largest prompt 1,381 tokens).

| Model | Easy (48) | Standard (72) | Hard (74) | All (194) | Latency p50 |
|---|---|---|---|---|---|
| chance | 0.28 | 0.31 | 0.36 | 0.32 | - |
| `tev1:4b` | 1.00 (1.00) | 0.94 (0.88) | 0.57 (0.55) | 0.81 (0.79) | 196 ms |
| `tev1:0.8b` | 1.00 (0.96) | 0.69 (0.66) | 0.41 (0.38) | 0.66 (0.63) | 68 ms |
| **`pacman-0.8b`** | 0.79 (0.37) | 0.39 (0.34) | 0.39 (0.37) | 0.49 (0.36) | 69 ms |
| Jev 1.13.0 (JevBench's published run) | 1.00 | 0.99 | 0.73 | 0.89 | - |

Accuracy above chance, as a share of the room above chance ((accuracy - chance) / (1 - chance)), is 0.73 for `tev1:4b`, 0.50 for `tev1:0.8b`, and 0.24 for `pacman-0.8b` over all 194 items. The Jev row is computed from JevBench's published per-item outcomes (v1.2) on the same 194 items, not measured here.

On the 40-scenario Pac-Man probe (section 4), rerun on the same day, the order reverses:

| Model | Probe accuracy | Probability on the correct move | Mean top probability |
|---|---|---|---|
| `tev1:4b` | 0.88 | 0.64 | 0.69 |
| `tev1:0.8b` | 0.42 | 0.40 | 0.54 |
| **`pacman-0.8b`** | **0.93** | 0.52 | 0.52 |

**Where the loss is.**

- Paired with `tev1:0.8b` item by item, `pacman-0.8b` gets 48 items wrong that `tev1:0.8b` gets right and 15 right that it gets wrong. On easy items the count is 10 lost and none gained; on standard items, 26 lost and 4 gained.
- On the standard tier `pacman-0.8b` is near chance in every family: intent 0.25, routing 0.33, ordinal 0.33, policy 0.42, extraction 0.50, against 0.58-1.00 for `tev1:0.8b`. The exception is judging whether an answer is adequate, where it scores 0.50 against 0.33. On easy items it still picks the right tool on 12 of 12 and the right intent on 7 of 12.
- The hard tier does not separate the two 0.8B models. Both are within a few points of chance (0.41 and 0.39), so specialization had little to lose there. Only `tev1:4b` is clearly above chance on it.
- `pacman-0.8b`'s answers lean toward the first two options. Options A or B are correct on 120 of the 194 items, and it picked A or B on 166. On yes/no items it said yes 85% of the time, against 47% in the labels and 66% for `tev1:0.8b`.
- Its probabilities stay close to uniform. When it is right on easy items, it is right by a small margin: 0.79 accuracy with 0.37 on the correct option. `tev1:0.8b` puts 0.96 there. Part of this is how it was trained: the oracle's soft targets spread probability over near-equal moves, so even on the Pac-Man probe its top option averages only 0.52. Out of domain the spread is larger still, and its probabilities are not useful as confidence.

**What the comparison does and does not show.** `pacman-0.8b` was not made by further training `tev1:0.8b`. It is a LoRA on the Qwen3.5-0.8B base, because Tev1's 0.8B weights have only been seen as Ollama GGUF. Tev1 is itself a LoRA on Qwen3.5. The two models are best read as two adapters on the same family of base model: one trained on general decisions, one only on Pac-Man. The gap above is therefore the general decision training `pacman-0.8b` never received, together with anything the Pac-Man training took from the base model. The base model is not served in Ollama here, so the two parts are not separated. Its remaining skill on easy items probably comes from the base model, which is not tested. Other limits: the set is small (12 items per family), it was run once (answers are deterministic), and it excludes the longest hard items.

The practical reading is that `pacman-0.8b` replaces a general decision model rather than extending one. It is the better choice when the model only ever plays this game, and the worse one when the same model must also route tickets or check policies. Keeping both would mean serving two adapters, or training the Pac-Man data together with general decisions, which was not tried.

### 16. Lookahead for a closed model

Section 13's lookahead reached Qwen through its prompt, so nothing about it needs open weights. The `features-peek5s` encoder puts the same facts into the `/v1/systemone` state, so any decision model can receive them, including a hosted one. It is the features encoding plus `lookahead_5s`: for each option, the outcome of `peek()` (play the option, then greedy moves for 5 s) as seconds until death or null, pellets, power pellets, ghosts eaten, and points. The instructions add one paragraph with section 13's wording: what the rollout is, that frightened ghosts turn at random, that it is one possible future, and what each field means. Like every encoder, it reports outcomes and never says which option is better.

Seeds 100-109, realtime clock, 1x speed, 5-minute cap. Both models play the same encoder; only the model differs.

| Player | Encoder | Mean pellets | Mean score | Mean survival | Mean level reached | Decision p50 | Stale answers | Input tokens per decision |
|---|---|---|---|---|---|---|---|---|
| Jev (section 12) | features | 178 | 2,482 | 52 s | 1.0 | 115 ms | 3.0% | 609 |
| **Jev** | features-peek5s | **567** | **9,150** | 175 s | 2.5 | 111 ms | 2.2% | 865 |
| `tev1:4b` (section 11) | features | 148 | 1,660 | 55 s | 1.0 | 202 ms | 10.7% | 368 |
| `tev1:4b` | features-peek5s | 212 | 2,907 | 68 s | 1.0 | 260 ms | 8.6% | 578 |
| `oracle-5s` (section 10) | features | 1,109 | 19,789 | 300 s (cap) | 4.7 | 5 ms | 0% | - |
| Qwen3.8-27B, lockstep (section 13) | features + lookahead in the prompt | 1,396 | 22,054 | 299 s | 6.0 | 309 ms | - | 471 (chat prompt) |

Per game, Jev with lookahead ate 432, 730, 481, 418, 972, 966, 301, 671, 461, and 234 pellets. Every game ended with all three lives lost; two (seeds 104 and 105) lasted to about the 5-minute mark and reached level 4. `tev1:4b` with lookahead ate between 171 and 237 pellets and never cleared level 1.

**Jev gains more than threefold.** The same closed model, with no change but its input, went from 178 to 567 pellets and from 52 s to 175 s. That is above the greedy rule (201) and the fine-tuned `pacman-0.8b` on features (456). Its latency did not change: the 256 extra tokens cost the hosted service nothing measurable.

**It reads the facts, imperfectly.** In 5,760 states where at least one option died in its rollout and at least one survived, Jev chose a dying option 12.8% of the time (`tev1:4b`: 16.8%). These are mostly deaths far down the rollout: Jev chose an option that died within 1.5 s only 20 times. In 441 of its 739 dying choices, that option promised more points than every surviving one. Its probability on those choices was lower (mean confidence 0.43 against 0.65). When every option died, it took the latest death 65% of the time. The rollouts themselves are pessimistic, because they play greedy after the first junction; before most deaths Jev had spent several decisions in states where every rollout died.

**`tev1:4b` gains little.** 148 to 212 pellets is a 43% gain, against more than 200% for Jev and about 360% for Qwen (lockstep). Part of the gap is latency: the extra 210 tokens took its p50 from 202 ms to 260 ms, and an answer applies about eight ticks after the state it was asked about. The rest is how the model reads the same facts.

**Cost.** The 10 Jev games used 9.72 million input tokens over 11,231 decisions, about $0.41 at the $0.042 per million seen on Vercel AI Gateway (section 12, conclusion 7). Games last three times longer, so the run cost about five times the features-only run.

**What this says.** The input knob is open to a closed model, and it took Jev past the fine-tuned student: Jev with lookahead (567) outplays `pacman-0.8b` on plain features (456). How much the knob is worth depends on the model: the same facts gave `tev1:4b` 43%, Jev about 220%, and Qwen about 360%. And for Jev the facts are worth less than a fixed formula over them: `oracle-5s` scores the same rollouts and averages 1,109 pellets. Once the input contains the outcomes, a model earns its place only by weighing them better than a formula does, which Qwen did and Jev did not.

**Caveats.** Qwen's numbers are lockstep, where the game waits for each answer; Jev's and `tev1:4b`'s are realtime. The rollouts run inside the encoder, before the request is timed, so their 5-10 ms of CPU is not charged to game time as the oracle's is. One run per seed; Jev's games vary from 234 to 972 pellets. `tev1:4b` ran on local Ollama with no other games running.

## Conclusions

1. Use `tev1:4b` for the real-time demo. At about 160 ms, a decision is shorter than the typical 400 ms to 1.2 s Pac-Man needs to reach the next junction at arcade speed.
2. Use a 0.8B model as the fine-tuning target and `tev1:0.8b` as its off-the-shelf baseline. The fine-tuned `pacman-0.8b` beats every off-the-shelf model and the scripted rule in play, at `tev1:0.8b`'s latency.
3. Label training data with `oracle-5s`. It is far stronger than any model we tried and costs milliseconds of local CPU per label. It also scores every option, so labels can be soft: ties become visible instead of being broken arbitrarily. Keep Qwen3.8-27B only as a comparison point.
4. Mind the information gap. The oracle sees the full game state, and the student sees only its encoding. Decisions that depend on facts the encoding drops cannot be learned from it. Measure the student against the oracle on held-out states to see how large this gap is.
5. Invest in the encoder before the model. Keep states compact and factual, and measure every encoder change in tokens as well as accuracy.
6. Compare clocks before comparing players. All Qwen results are lockstep, where the game waits for each answer; Jev, the Tev1 models, and `pacman-0.8b` are realtime, where slow answers arrive after the junction has passed. Qwen without thinking (301 pellets) beating Jev (178) does not yet show it would at realtime.
7. Do not argue cost against Jev. At the price seen on Vercel AI Gateway ($0.042 per million input tokens), a whole game of about 340 decisions at 605 tokens each costs under a cent. The case for a fine-tuned local model is task accuracy, latency, offline use, and control. The cost case holds against calling a large general model at every step.
8. A stronger teacher may not make a stronger student. The student's ceiling is set by what it sees at inference time. Qwen with lookahead beats the oracle by reading rollouts the features-only student never sees, so relabeling with it may leave the student near its current plateau. A student given the same lookahead facts would test this; the rollouts take milliseconds of CPU.
9. Treat a plain chat model with constrained JSON as the off-the-shelf baseline. On this task it matches or beats the decision models (Jev, `tev1:4b`) from the same facts, so the decision API is a convenience here, not a capability. `phi4-mini` is the fastest and strongest of those tried and returns probabilities. The fine-tuned `pacman-0.8b` still doubles it.

## Possible follow-ups (not planned)

The project stops here: the demo sequence (off-the-shelf 0.8B, off-the-shelf 4B, fine-tuned 0.8B) is complete. If it resumes, the gap to the oracle is the obvious target:

- Enrich the encoder: ghost identities, the second-nearest ghost, and longer-range pellet counts.
- Run a DAgger round with states from `pacman-0.8b`'s own games.
- Try a full-map student: fine-tune on `ascii-full` with oracle labels and compare.
