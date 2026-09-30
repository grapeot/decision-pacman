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

## Lessons Learned

- Latency scales with input tokens (~0.8 ms per token for nimble on M3 Ultra). Putting the static maze first and the dynamic part last saved only ~35 ms, so prefix caching does not make large states cheap. Exact repeats of a request return in ~35–60 ms, so benchmarks must use fresh states or they will look far faster than a real game.
- On the full ASCII board, nimble's confidence stayed between 0.00 and 0.09. On per-direction facts it chose the obvious move with probability ~0.99. The representation matters more than the model size between nimble and tev1:4b.
- tev1:0.8b is fast but near chance on simple situations. Do not make it the default.
- The models fled frightened ghosts. Game rules have to be stated in `instructions`, because the model does not bring Pac-Man knowledge into the decision.
- Ollama's launch blog includes its own nimble Pac-Man example (91 ms per decision on an M5 Max, only legal moves as options). It is a recorded, turn-based replay (JSON in the page), with no code to reuse.
- The synthetic scenarios have one clearly correct move each. They test format and basic judgment, not play on real, ambiguous game states. Teacher and student quality claims need states sampled from the engine.
- The teacher server rejects `response_format: json_schema` (no constrained decoding), so labels come from parsing plain text. A prompt that asks for JSON was 100% parseable in 120 calls.
