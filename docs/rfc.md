# RFC: Architecture and key decisions

## Summary

The game is a pure browser application. A deterministic TypeScript engine steps at a fixed 30 Hz. An asynchronous agent loop, decoupled from the simulation, snapshots the state, encodes it as compact JSON, and queries a local `/v1/systemone` endpoint (Ollama by default) directly from the browser. The answer becomes a buffered direction intent that the engine applies at the next legal tile. There is no application server. The engine has no DOM dependency, so the same code runs headless in Node for tests, benchmarks, teacher data generation, and evaluation.

```
                    browser tab
 ┌──────────────────────────────────────────────────────┐
 │  requestAnimationFrame ─► Renderer (canvas) ◄─┐       │
 │                                               │ reads │
 │  30 Hz fixed-step ─► Engine.step(state, intent)       │
 │                           ▲           │ snapshot      │
 │                    intent │           ▼               │
 │                      Agent loop ─► Encoder ─► JSON    │
 │                           ▲                     │     │
 └───────────────────────────┼─────────────────────┼─────┘
                             │ answer + probs      │ POST /v1/systemone
                             └──── decision model endpoint ◄┘
                                   (Ollama :11434 or ollaya :11435)
```

Later stages reuse the engine headless:

```
 engine (Node) ──states──► teacher (OpenAI-compatible chat, lockstep) ──labels──┐
      │                                                                          ├─► dataset (JSONL)
      └──states──► lookahead oracle (engine rollouts) ────────labels─────────────┘
 dataset ──► LoRA fine-tune (Qwen3.5-0.8B base, RTX 5090) ──► GGUF ──► /v1/systemone ──► evaluation
```

## Constraints from measurement

Full method and numbers are in `docs/model_evaluation.md`. The constraints that shaped this design:

1. **Latency is proportional to input tokens.** Decisions are prefill-bound: about 0.8 ms per input token for `nimble` on an M3 Ultra. Prefix caching saved only about 8% on a large state, and q4 quantization gave no speedup. The only effective lever is a smaller state.
2. **Representation matters more than model size.** On a full ASCII board, `nimble` was close to guessing (confidence 0.00–0.09). On per-direction facts, `tev1:4b` and `nimble` chose clear moves with probability around 0.99. A plain-English rendering of the same facts did not help.
3. **Off-the-shelf model choice.** `tev1:4b` was the best trade-off: about 160 ms and 0.88 on the probe. `tev1:0.8b` ran at about 65 ms but scored 0.40, near chance. `nimble` was slower (270 ms) and less accurate (0.68).
4. **The browser can call Ollama directly.** Ollama echoes `Access-Control-Allow-Origin` for `http://localhost:*` origins.
5. **Qwen3.8-27B works as a teacher.** Thinking off: 270 ms, 100% parseable, 0.97 on the probe, about 16 labels per second at concurrency 8. Thinking on: 1.1 s and 1.00. The serving stack does not support constrained decoding, so the teacher's answers are parsed from plain text.

## Components

- `src/engine/`: maze, movement, ghost AI, pellets, scoring, lives, and level state. It exposes `createGame(seed, options)` and `step(state, intent)` as pure functions over plain data, with a seeded RNG. There are no timers and no DOM access. Events go into an array on the state.
- `src/encoders/`: `(state, decisionPoint) => { state, instructions, options }`. Encoders compute facts with BFS over the maze and never call a model.
- `src/agent/`: the agent loop, the `/v1/systemone` client, the decision log, and the policy interface shared by model, teacher, human, random, and scripted policies.
- `src/oracle/`: the lookahead oracle. It scores each option by rolling the engine forward.
- `src/render/`, `src/ui/`: Canvas 2D drawing, HUD, controls, and error panel. They read engine state and never change it.
- `scripts/`: headless runner, teacher data generation, dataset export, probes, recording, and dev/build/test entrypoints.
- `training/`: Python (uv) scripts for LoRA fine-tuning, evaluation of the checkpoint, and GGUF export. They run on the GPU machine.
- `ios/`: a SwiftUI app that serves the web build in a WKWebView and answers its decisions with llama.cpp on device or, for Jev, through a native HTTPS bridge (decision 10).

## Key decisions

### 1. Pure frontend, browser calls the model endpoint directly

The browser fetches `<base>/v1/systemone` itself. A server would add a process and a network hop and buy nothing, because the model is already a local HTTP service. The base URL comes from `VITE_DECISION_BASE_URL` and can be changed in the UI. In development, Vite proxies `/decide/*` to the configured endpoint, which sidesteps CORS for other ports and origins. For a static build on another origin, the viewer adds that origin to `OLLAMA_ORIGINS`. The error panel detects each failure and shows the fix.

### 2. Fixed 30 Hz simulation, decoupled from rendering and from the model

The engine advances in fixed 1/30 s steps through an accumulator driven by `requestAnimationFrame`. Rendering draws the latest state every animation frame. The agent loop runs on its own async schedule. Nothing in the simulation awaits a network call, so a slow or failed request only means Pac-Man keeps following the last intent. Fixed steps keep the engine deterministic, which makes replays, tests, and headless runs reproducible.

### 3. Answers become planned turns or an immediate reversal

An exit answer becomes a turn planned for that junction tile (`StepInput.turn`), which the engine takes when Pac-Man reaches the tile center. A "back" answer turns Pac-Man around at once (`StepInput.reverse`). The two are kept apart on purpose. After a corner, a junction exit can point opposite to the current heading, and treating it as a reversal made Pac-Man dither. Keyboard play uses an untargeted buffered direction (`StepInput.intent`) with arcade semantics. Between decisions Pac-Man keeps going and follows single-exit corners, so a stale or missing answer costs a missed turn, never a stop in a corridor.

### 4. Decisions are about the next junction, with latency-sized lookahead

Real choices happen at junctions. Each request asks about the next junction on the current heading, with options set to that junction's exits plus "back". The agent keeps a moving average of decision latency and turns it into a lookahead distance (tiles Pac-Man covers while an answer is in flight, plus a margin). Suppose Pac-Man is closer to the next junction than that distance, and a turn there is already planned. Then the agent asks about the junction after it, because a new answer for the near one would arrive too late. An answer is discarded if its junction is no longer ahead. The HUD reports the stale rate. With lookahead, tev1:4b's stale rate fell from 22% to 10%.

### 5. Encoders report facts, never verdicts

An encoder that says "left is dangerous" turns the model into a rubber stamp on a scripted policy. An encoder that sends the raw board makes the model guess. The default `features` encoder sends, for each option, the BFS distance to the nearest ghost through that exit and whether it is frightened or approaching, pellets within N steps, distance to the nearest power pellet, and whether the path dead-ends within N steps. Global facts are frightened time left, pellets left, and lives. It never sends a safety label, a recommended move, or a per-option score. Game rules go into `instructions`. An `ascii-window` encoder (a crop around Pac-Man) and an `ascii-full` encoder exist for comparison and can be switched live.

This rule matters even more for fine-tuning. If the features contained verdicts, the fine-tuned model would learn to copy them, and the improvement would say nothing about the model.

### 6. The token budget is the latency budget

The `features` encoder targets about 200 tokens: short keys, no pretty-printing. Every encoder change records its input token count next to its score.

### 7. Models

The default is `tev1:4b` on Ollama. `tev1:0.8b` is the fine-tuning baseline. `nimble` and ollaya models remain selectable. The app sends `keep_alive: -1` during a game and one warm-up request at start.

### 8. Stack: Vite + TypeScript + Canvas 2D, no UI framework

The game is a canvas plus a small HUD, so a UI framework would add a second render cycle for no benefit. Vite provides the dev server, the proxy, and the static build. Vitest runs engine and encoder tests. The headless runner and data scripts are TypeScript run with `tsx`, importing the same engine and encoders as the browser. Training is Python, because the fine-tuning stack is.

### 9. Engine fidelity: faithful in behavior, not in cycles

The engine uses the classic 28×31 tile layout, sub-tile movement, the four ghost targeting rules, timed scatter and chase phases, frightened mode with seeded random turns, the ghost house, and the tunnel. Speeds are level 1 arcade values in tiles per second, scaled by a speed multiplier. Frame-exact speed tables and arcade bugs are out of scope. Graphics are drawn in code.

### 10. iPhone demo: one page, three players behind native bridges

The app runs the same web game and the same prompts as the browser. The page renders each prompt and native code runs it, so prompt text has one implementation (TypeScript, unit-tested) and the Swift side stays small. The fine-tuned 0.8B is scored from option-letter logits, as Ollama's `/v1/systemone` does. phi4-mini is asked as `llm:phi4-mini` asks Ollama, with a GBNF grammar in place of Ollama's JSON schema, and checked against Ollama's recorded answers on the Mac. Jev goes through native code because the page's `app://local` origin cannot call the API directly and the key must stay out of the page; the endpoint is fixed in Swift so the page cannot send the key elsewhere. Only one on-device model is loaded at a time. Model files and the key are copied into the app's Documents, never bundled.

## Headless runner

`scripts/run_headless.ts` plays N seeded games with any policy and writes one JSONL record per decision plus a per-game summary (score, pellets eaten, ticks survived, deaths, decisions, latency percentiles). Two clocks:

- `lockstep`: the simulation waits for each decision. It measures decision quality without latency and is how the teacher plays.
- `realtime`: each decision's measured latency converts to simulation ticks, which advance before the answer applies. It reproduces browser timing without rendering and is the clock for evaluation.

Baselines: `random` (uniform over legal options) and `greedy` (nearest pellet, avoid non-frightened ghosts within k steps).

## Stage 2: teacher data

**Labeler.** `oracle-5s` (`src/agent/oracle.ts`). For each option it copies the game, plays the option, and rolls the real engine forward 5 seconds, with greedy play at later junctions. Ghost moves follow the real rules, so the lookahead is exact except for frightened ghosts' random turns. The value of an option is the score gained, minus a large penalty for a death that shrinks the later the death happens, minus the distance to the nearest pellet at the end. It costs about 5 ms of CPU per state. A chat model (Qwen3.8-27B) was evaluated first and dropped as the labeler: it plays far worse than the oracle and costs a remote GPU (`docs/model_evaluation.md`, experiments 9 and 10).

**Soft labels.** The record stores every option's value, not only the argmax. Training can then turn values into a target distribution, so near-ties no longer force an arbitrary hard label.

**State sampling.** If only the oracle plays, the data covers only the positions a strong player reaches. The fine-tuned student will then wander into positions the oracle never saw. States are therefore sampled from a mix of rollouts: the oracle playing, greedy and random play, and in round two the fine-tuned student playing (DAgger-style). All sampled states are labeled by the oracle, whoever produced them.

**Information gap.** The oracle sees the full game state, and the student sees only its encoding. Stage 4 measures agreement with the oracle on held-out states. A large gap points to the encoder, not the model.

**Record format.** One JSONL line per state: seed, tick, source policy, encoder, encoded state, instructions, options, per-option values, and the label.

**Splits.** Train, validation, and evaluation use disjoint seed ranges: evaluation 100-199, validation 900-999, training 1000 and up. `scripts/gen_teacher_data.ts` refuses evaluation seeds.

## Stage 3: fine-tuning

**Prompt parity with Ollama.** Ollama builds the scoring prompt in `decision/systemone.go`. It sends one user message, `{"context": <state>, "schema": [{"name", "description", "choices": [{"code": "A", "value", "description"}, ...]}]}` followed by `\n\nRequested field: "move"`, after the model's system prompt. The chat template is applied with thinking off. It then reads the logits of the option letters at the next position and takes a softmax over those letters only. `training/prompt.py` reproduces this byte for byte, including Go's HTML-safe JSON escapes and Tev1's system prompt. On 50 validation states, the served Q8_0 model and the bf16 training weights agree to within 0.027 in probability (median 0.010), and pick the same option on 48 of 50. The two that differ are near-ties.

**Base model.** `Qwen/Qwen3.5-0.8B`, the base of `tev1:0.8b` (Ollama reports 752M parameters). Tev1's own 0.8B weights are published only as GGUF.

**Data.** `training/build_sft.py` turns oracle-labeled states into rows with the Ollama messages and a soft target: softmax of each option's oracle value divided by a temperature of 50 points (about five pellets). Exact duplicate states are removed. v1 data: 31,585 training rows (oracle 17,482, greedy 8,075, random 6,028 as players) and 4,975 validation rows from separate seeds.

**Training.** `training/train.py`: LoRA (rank 16, alpha 32) on every language-side linear layer, lr 1e-4 cosine with 3% warmup, batch 32, 2 epochs (1,975 steps), bf16, one RTX 5090. The loss is cross-entropy between the soft target and the softmax over option-letter logits at the answer position. Only that position is projected through the output layer, so memory stays small despite the 248k-token vocabulary. A startup check compares that projection with the model's own logits. Training takes about 40 minutes.

**Serving.** `training/export_gguf.sh` merges the adapter and converts to Q8_0 GGUF with llama.cpp. The GGUF is imported through Ollama's `/api/create` with Tev1's system prompt, as `pacman-0.8b`. Ollama 0.35.0 does not accept `CAPABILITY` in a Modelfile, but it serves `/v1/systemone` for the imported model anyway. The game needs no change beyond the model name.

**GPU use.** The GPU host serves the teacher model on all three cards. Training borrows one card with the cluster's lease script, which drops that replica from the router, and then returns it. The other two replicas keep serving.

## Stage 4: evaluation

Every model plays the same held-out seeds in the `realtime` clock at the same game speed on the same machine: `tev1:4b`, `tev1:0.8b`, the fine-tuned 0.8B, the teacher (lockstep, as an upper bound), `greedy`, and `random`. The metrics are pellets eaten, score, ticks survived, deaths, decision latency, and agreement with the teacher on held-out states. Results and plots go to `docs/model_evaluation.md`.

## Risks

- **Imported model not accepted by Ollama's decision runner.** Mitigation: the fallbacks in Stage 3. This is verified early with an untrained export, before any real training run.
- **Teacher labels are weaker on real positions than on the probe.** The probe scenarios have one clearly correct answer. Real positions are ambiguous. Mitigation: the oracle cross-check, and a hand review of disagreements before training.
- **Distribution shift.** The student reaches states the teacher never did. Mitigation: mixed-policy sampling and a DAgger round.
- **Shared GPU host.** Teacher serving and training compete for the same GPU. Mitigation: run the stages in sequence.

## Error handling

The client distinguishes connection refused, CORS rejection, model not found, and 400/413 responses, and shows a specific fix for each. On any failure the game keeps running on the last intent, and the agent retries with backoff.

## Alternatives considered

- **Server-side game with a WebSocket viewer.** It adds a runtime and a protocol for a single-player demo whose model is already a local HTTP service.
- **Turn-based play in the browser.** It scores higher but removes the point of the demo. It survives as the `lockstep` clock.
- **Chat model as the in-game player.** Qwen3.8-27B is accurate but needs parsing and a remote GPU, and gives no calibrated probabilities. It is used offline as the teacher instead.
- **Oracle-only labels.** They are free and exact for the rollout horizon, but they depend on the rollout policy for ghosts and on the horizon. They are kept as a cross-check. If agreement shows the oracle is the better labeler, the dataset can switch to it without other changes.
