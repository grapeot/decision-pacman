# Working notes

## Changelog

### 2026-09-30

- Scaffolded the project: docs, AGENTS.md, .gitignore, .env.example, and empty src/scripts/tests.
- Renamed the project from nimble_pacman to decision_pacman after the default model moved to tev1:4b.
- Rewrote the PRD and RFC around four stages (real-time game, teacher data, fine-tuning a 0.8B model, evaluation), and moved measurements into `docs/model_evaluation.md`.
- Verified the local Ollama 0.35.0 `/v1/systemone` endpoint with `nimble`, `tev1:4b`, and `tev1:0.8b` on an M3 Ultra.
- Measured latency for the full ASCII board (~500 tokens: nimble ~430 ms, tev1:4b ~240 ms, tev1:0.8b ~80 ms) and for per-direction feature JSON (~200 tokens: nimble ~220–260 ms, tev1:4b ~150–170 ms, tev1:0.8b ~55–75 ms).
- Confirmed that Ollama returns CORS headers for `http://localhost:*` origins, so the browser can call it directly.
- Wrote the PRD, RFC, and test strategy.
- Added `scripts/probe_decision_models.py`: 40 fresh synthetic scenarios with a known correct move per model. Results on json state: nimble q8 270 ms / 0.68, tev1:4b 156 ms / 0.88, tev1:0.8b 67 ms / 0.42, laya:en (ollaya) 20 ms / 0.40.
- Quantization variants on the same probe: nimble:9b-q4_K_M 286 ms / 0.60, tev1:4b-q4_K_M 161 ms / 0.88, tev1:0.8b-bf16 65 ms / 0.40. q4 buys no speed on M3 Ultra, because short-output decisions are prefill-bound (compute), not bandwidth-bound.
- Tested a plain-English state against compact JSON: no gain (laya 0.40 to 0.47, tev1:4b 0.88 to 0.78). Kept JSON.
- Installed ollaya 0.7.5 (port 11435) to benchmark Kev, decider, laya, and winnow. kev:0.8b ran at 729 ms / 0.40 on the Mac (CPU path).
- Confirmed Tev1 is a plain LoRA on Qwen3.5 with an MIT-licensed recipe. Its 4B weights are on Hugging Face. The 0.8B weights have been seen only as Ollama GGUF, so fine-tuning starts from the Qwen3.5-0.8B base.
- Added `scripts/probe_teacher.py` and measured Qwen3.8-27B (NVFP4, OpenAI-compatible server) as a possible distillation teacher on the same scenarios. Thinking off: p50 270 ms, 40/40 parseable, 40/40 legal, accuracy 0.97. Thinking on: p50 1.1 s, accuracy 1.00, ~190 output tokens. Thinking off at concurrency 8: 16 labels/s.

### 2026-09-30 (Stage 1)

- Built the engine (`src/engine/`): classic maze, fixed 30 Hz step, four ghost targeting rules, scatter/chase schedule, frightened mode, ghost house, tunnel, lives, and level clear. Deterministic from a seed.
- Added decision points (`findDecisionPoint`): the next junction on the current heading, following single-exit corners automatically.
- Added the `features` and `ascii-window` encoders, the `/v1/systemone` client, random/greedy/model policies, and the browser agent loop.
- Added the canvas renderer and HUD (probability bars, latency, stale rate, ticks per second, the encoded state the model sees) and keyboard play.
- Added `scripts/run_headless.ts` (realtime and lockstep clocks, JSONL logs under `runs/`) and `scripts/record_demo.ts` (Playwright recording converted to MP4).
- Headless, 1x speed, realtime clock, seeds 100+: random 30.3 pellets (10 games), greedy 200.7 (10 games), tev1:4b 168.7 (3 games, p50 207 ms, 10% stale).
- Recorded `docs/media/demo_tev1_4b.mp4` (30 s, seed 100): 1250 points and 101 pellets, one life lost near the end.
- Tests: 28 passing (engine, planned turns, decision points, encoders).

### 2026-09-30 (Stage 2 smoke test)

- Moved the game-driving loop into `src/sim/runner.ts`, shared by the headless runner and data generation. Greedy results were identical before and after (10 games, 200.7 pellets).
- Added the teacher policy (`src/agent/teacher.ts`): an OpenAI-compatible chat model sees the student's encoded state and answers `{"move", "reason"}`. An unparseable or illegal answer drops the state.
- Added `scripts/gen_teacher_data.ts`: a player drives the game in lockstep and a labeler labels every decision state. Records use schema `decision-pacman/teacher-v1` under `data/` (gitignored). Seeds 100-199 are refused.
- Smoke test (player and labeler both Qwen3.8-27B, thinking off, seed 1000, 30 s of game time): 144 labeled states in 46 s wall time, 0 failures, label latency p50 301 ms and p90 368 ms, 3.1 labels per second sequential. The teacher scored 2470 with 155 pellets and 1 death in those 30 s. Labels: left 53, up 35, down 25, right 22, back 9. It agreed with the greedy rules on 70% of states. The disagreements sampled were ties, which the teacher broke toward a power pellet.
- Tests: 33 passing (adds teacher reply parsing and prompt tests).

### 2026-09-30 (teacher comparison and search baseline)

- Added the `ascii-full` encoder: the whole board with row and column numbers, plus positions in words (about 1,350 characters).
- Qwen3.8-27B as a player, lockstep, 30 s, seeds 1001-1003: features, thinking off: 169 pellets and 2 deaths in total. Features, thinking on: 176 pellets, 0 deaths, 13 unparseable answers out of about 500. ASCII full map, thinking off: 99 pellets, 6 deaths. ASCII full map with thinking was stopped after 48 minutes, too slow to matter.
- Qwen misreads the ASCII map. Sampled reasons mention dead ends (the maze has none), a tunnel on row 29 (it is on row 14), and a far power pellet when one was on the same row.
- Added `src/agent/oracle.ts`: for each option, copy the game, play the option, and roll the real engine forward with greedy play at later junctions. The score gained is penalized heavily for a death, less for a later one.
- Oracle results on evaluation seeds 100-109, realtime, 5-minute cap: `oracle-3s` averaged 1,009 pellets (about 4 levels) and 16,546 points, 3 ms per decision. `oracle-5s` averaged 1,109 pellets and 19,789 points, 5 ms. `oracle-8s` averaged 1,011 pellets and 15,858 points, 7 ms. All 30 games reached the time cap. A longer horizon is not better past 5 s, probably because the greedy rollout policy gets less reliable further out.
- 5-minute lockstep head-to-head, seeds 100-102: `oracle-3s` ate 971, 1,178, and 1,211 pellets with 1 death per game. Qwen (features, thinking off) ate 361, 361, and 180 pellets and lost all three lives within 74, 93, and 39 seconds.
- A literature survey agreed: the strongest simple agents rely on an exact ghost model and a safe-path test, which this engine provides for free.
- Fixed output directories colliding when two runs start in the same millisecond. Names now include the encoder and the process id.

## Lessons Learned

- Latency scales with input tokens (~0.8 ms per token for nimble on M3 Ultra). Putting the static maze first and the dynamic part last saved only ~35 ms, so prefix caching does not make large states cheap. Exact repeats of a request return in ~35–60 ms, so benchmarks must use fresh states or they will look far faster than a real game.
- On the full ASCII board, nimble's confidence stayed between 0.00 and 0.09. On per-direction facts it chose the obvious move with probability ~0.99. The representation matters more than the model size between nimble and tev1:4b.
- tev1:0.8b is fast but near chance on simple situations. Do not make it the default.
- The models fled frightened ghosts. Game rules have to be stated in `instructions`, because the model does not bring Pac-Man knowledge into the decision.
- Ollama's launch blog includes its own nimble Pac-Man example (91 ms per decision on an M5 Max, only legal moves as options). It is a recorded, turn-based replay (JSON in the page), with no code to reuse.
- The synthetic scenarios have one clearly correct move each. They test format and basic judgment, not play on real, ambiguous game states. Teacher and student quality claims need states sampled from the engine.
- The teacher server rejects `response_format: json_schema` (no constrained decoding), so labels come from parsing plain text. A prompt that asks for JSON was 100% parseable in 120 calls.
- Keep "turn around now" separate from turns. When a junction exit points opposite to the current heading (after a corner), treating it as a reversal made Pac-Man dither in place. `StepInput.reverse` turns around, and `StepInput.turn` plans a turn for one specific junction tile.
- Features must count the corridor between Pac-Man and the junction. Measuring only from the junction outward hid pellets and ghosts on the way and made "back" look closer than it was.
- Latency has to become lookahead. Asking about a junction Pac-Man reaches before the answer returns wastes the call. The agent now asks about the junction after next once the near turn is planned and Pac-Man is within the latency distance. Stale answers fell from 22% to 10%, and pellets rose from 138 to 169.
- In the browser, wait one tick after queuing an answer before asking again. Otherwise the next request is planned from a state without the new turn. This alone took the browser stale rate from about 40% to 13%.
- Do not ask during frozen phases (ready, dying). Identical states hit the server's cache and pull the latency p50 down to about 40 ms, which misrepresents the real 200 ms.
- The greedy baseline scored 238 pellets without lookahead and 201 with it, because it decides earlier on older information. Compare baselines only within the same agent setup.
- Many teacher states are ties: no ghost nearby and equal food on several exits. A hard label on a tie is noise for the student. Before training, either down-weight ties, keep the teacher's reason to filter them, or ask for a distribution instead of a single move.
- Sequential lockstep labeling runs at about 3 states per second. 20,000 states would take about 2 hours on one game. Several games in parallel against the shared teacher endpoint cut that, within a concurrency cap.
- The engine is a perfect simulator, so rollout search beats any model we tried by a wide margin, and it runs locally in milliseconds. Label training data with the oracle, not an LLM. Keep the LLM as a comparison point.
- 30-second windows hide the difference between players. They are mostly pellet collection early in the level. Survival only separates over minutes.
- zsh does not word-split `$var`. `set -- $cfg` in a loop passed both arguments as one. Use functions with explicit arguments.
