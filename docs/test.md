# Test strategy

## Unit tests (Vitest, offline, default `npm test`)

- Engine: movement and wall collision, buffered intent (turn at the first legal tile, immediate reverse), tunnel wrap, pellet and power pellet scoring, frightened timer, ghost eaten and death transitions, level clear.
- Ghost AI: each ghost's target tile in scatter and chase for fixed positions, and no reversal except on mode change.
- Determinism: the same seed and intent sequence produce identical state after N ticks.
- Decision points: next-junction detection and the exit set for every junction on the maze.
- Encoders: output for fixed states is snapshot-tested, the options match the legal exits, the serialized size stays under the token budget (character count as a proxy), and no verdict fields (safe, danger, recommend) appear.
- Agent loop: with a fake client, at most one request is in flight, an answer for a stale junction is dropped, and a failed request leaves the intent unchanged.

## Integration tests (opt-in, need a running Ollama)

- `npm run test:live`: skipped unless `LIVE_DECISION_API=1`. It sends one request per encoder to each configured model and checks the response schema, that the chosen option is in the option set, and that the probabilities sum to about 1.
- `npm run probe`: prints latency and token counts per encoder and model. Record the results in `working.md` when models, encoders, or hardware change.

## Headless evaluation (manual, needs Ollama)

`npm run headless -- --model tev1:4b --encoder features --clock realtime --games 10 --seed 1` plays full games and writes JSONL under `runs/` (gitignored). A change to an encoder or to the agent loop is done when its summary is compared against the previous run and the `random` and `greedy` baselines on the same seeds, and the comparison is written to `working.md`.

## Manual browser check

With `npm run dev` and Ollama running: the tick rate reads 30, probability bars update, latency p50 matches the headless numbers for the same model and encoder, switching encoders mid-game works, and stopping Ollama mid-game shows the error panel while the game keeps running.

## Teacher data and fine-tuning

- Teacher adapter: offline tests with recorded responses cover parsing, illegal moves, and unparseable output (the state is dropped, never guessed).
- Dataset export: a schema check on every line, plus a check that no evaluation seed appears in train or validation.
- Prompt parity: before training, a script renders sample requests with the training formatter and compares them to the prompt the serving runtime builds. They must match exactly.
- Export smoke test: an untrained 0.8B export is served over `/v1/systemone` and answers the probe before any real training run.
- Fine-tuned model: `scripts/probe_decision_models.py` plus the held-out realtime games in `docs/model_evaluation.md`.

## E2E

None in v1. The canvas output is checked by hand. A Playwright smoke test (page loads, the game advances in human mode, the error panel appears without Ollama) is worth adding once the UI settles.
