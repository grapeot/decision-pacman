# PRD: Decision Pac-Man

## Goal

Show, end to end, what it takes to make a small local decision model play a real-time game well. The project has four stages:

1. **Real-time game.** A Pac-Man-style game runs in the browser at a fixed 30 ticks per second, and a local decision model steers it through Ollama's `/v1/systemone` API. The default model is `tev1:4b`. A recorded video shows it playing.
2. **Teacher data.** A large model (Qwen3.8-27B, served as an OpenAI-compatible chat endpoint) plays and labels game states in the same engine, without real-time pressure. This produces labeled data from real game states.
3. **Fine-tuning.** A 0.8B decision model is fine-tuned on that data on a single RTX 5090.
4. **Evaluation.** The fine-tuned model plays in the same engine against the off-the-shelf baselines.

The finished project doubles as a teaching sequence. An off-the-shelf 4B model plays, though not well. An off-the-shelf 0.8B model is fast but close to random. The fine-tuned 0.8B model is fast and plays well. The 0.8B size is the fine-tuning target for two reasons: it is closer to real time, and its low baseline leaves the most room to show what fine-tuning adds.

## Background

Ollama 0.35 added `POST /v1/systemone`, a decision API modeled on TypeSafe's Jev API. A request carries a `state` (text, or JSON serialized as text) and named questions (`choice`, `noul`, `score`), and every answer comes back with a probability for each option. Relevant local models are `tev1:4b` and `tev1:0.8b` (Together AI, LoRA fine-tunes of Qwen3.5 with an MIT-licensed training recipe) and `nimble` (9B, Bespoke Labs). The community runtime ollaya serves more open decision models over the same wire format.

The model evaluation in `docs/model_evaluation.md` drives the choices here. `tev1:4b` was the best off-the-shelf trade-off (about 160 ms, 0.88 on the probe). `tev1:0.8b` was fast (about 65 ms) but near chance (0.40). Qwen3.8-27B reached 0.97 with thinking off and 1.00 with thinking on, with fully parseable output. The probe scenarios are synthetic, so these numbers bound basic judgment only. Play quality is measured in the game.

Ollama's launch post shows a Pac-Man example, but it is a recorded, turn-based replay with no code. This project builds its own engine, frontend, encoders, data pipeline, and training.

## Users

- **Learner or demo viewer.** Watches the three models play in turn and sees latency, probabilities, and score side by side.
- **Tinkerer.** Clones the repo to try other models, encoders, prompts, or training data. Needs clear extension points and a headless runner with comparable metrics.

## Requirements

### Stage 1: real-time game (current)

- A playable maze game with pellets, power pellets, four ghosts with distinct chase behavior, frightened mode, lives, score, and level clear.
- A fixed 30 Hz simulation that never waits on the model.
- An agent loop that snapshots the state, encodes it, queries `/v1/systemone`, and hands the chosen direction to the game. At most one request is in flight, and the newest answer wins.
- Arcade-style control: the answer is a desired direction that the game applies at the first tile where the turn is legal.
- Options in each request are exactly the moves available at the decision point.
- A compact per-direction feature encoder (default) and at least one board-style encoder for comparison, switchable at runtime.
- A HUD with live probability bars, confidence, latency (last, p50, p90), decisions per second, and staleness.
- Model and endpoint selection, with `tev1:4b` on Ollama as the default. The endpoint is a base URL, so Ollama and ollaya both work.
- A game speed multiplier and keyboard play for humans.
- A clear on-page error when the endpoint is unreachable, blocked by CORS, or missing the model, with the exact fix.
- A headless runner that plays seeded games against the real API and writes JSONL per decision plus a per-game summary.
- A scripted screen recording of `tev1:4b` playing in the browser, committed as a short video or linked from the README.

### Stage 2: teacher data

- A teacher adapter that asks an OpenAI-compatible chat model for a move and parses a JSON answer. The teacher plays in `lockstep` mode, so the game waits for each answer.
- State sampling from a mix of policies (teacher, random, and later the student) so the data covers bad positions as well as good ones.
- Every sampled state is stored with the student-format encoding, the legal options, the teacher's label and reason, and an engine lookahead label for cross-checking.
- Throughput and concurrency limits that are configurable, because the teacher endpoint is shared.

### Stage 3: fine-tuning

- A training script for LoRA fine-tuning of a 0.8B Qwen3.5-based decision model on one RTX 5090, following the Tev1 recipe format.
- Held-out evaluation seeds that never appear in training data.
- An export path that serves the fine-tuned model over `/v1/systemone`, through Ollama if its decision runner accepts the model, otherwise through ollaya or a small scoring server.

### Stage 4: evaluation

- The same seeds and clocks for every model: off-the-shelf `tev1:4b`, off-the-shelf `tev1:0.8b`, fine-tuned 0.8B, the teacher, and random and scripted baselines.
- Results written to `docs/model_evaluation.md`.

### Won't have

- Cycle-accurate arcade behavior, original arcade art, audio, or branding.
- An application server for the game. The browser talks to the model endpoint directly.
- Multiple mazes or a full difficulty curve.

## Success criteria

1. **Stage 1:** `tev1:4b` drives the browser game at a steady 30 ticks per second with at least 4 decisions per second, the HUD shows live latency and probabilities, and a recording exists.
2. **Stage 1:** Over 10 seeded headless games in real-time mode, `tev1:4b` clearly beats a random-legal-move baseline in pellets eaten.
3. **Stage 2:** At least 20,000 labeled states. On a sample, teacher and lookahead labels agree often enough that disagreements can be reviewed by hand.
4. **Stage 4:** The fine-tuned 0.8B model stays under 100 ms per decision and matches or beats off-the-shelf `tev1:4b` in pellets eaten and survival on held-out seeds.

## Open questions

- **How much reasoning belongs in the encoder.** Encoders report facts (distances, counts, ghost states) and never verdicts such as "safe" or a recommended move. See RFC decision 5.
- **Default game speed.** At arcade speed Pac-Man covers about 7.5 tiles per second, so a 160 ms decision is a little over one tile stale. Pick the default after headless runs have measured score against speed.
- **Teacher as player or labeler.** The plan uses the teacher for both. The lookahead oracle may turn out to be a better labeler on tactical positions. The Stage 2 cross-check decides.
