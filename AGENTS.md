# AGENTS.md

A browser demo in which a local decision model (`/v1/systemone`, default `tev1:4b` on Ollama) plays a Pac-Man-style game in real time, plus a pipeline to generate teacher data, fine-tune a 0.8B model, and evaluate it. Read `docs/prd.md` and `docs/rfc.md` before changing architecture. Model choices and their evidence are in `docs/model_evaluation.md`. This repo targets public GitHub and is written in English.

## Structure

- `docs/prd.md`: goals, requirements, success criteria.
- `docs/rfc.md`: architecture, measured constraints, and key decisions. Update it when a decision changes.
- `docs/test.md`: what counts as verified.
- `docs/model_evaluation.md`: evaluation method, results, and conclusions. Add in-game results here.
- `docs/results.md`: current numbers and caveats. Update it when a headline number changes, and keep lookahead results out of the ladder.
- `docs/guides/`: how-to guides (run models, evaluate, distill, iPhone). Update them when a command or flag changes.
- `docs/working.md`: daily changelog plus lessons learned. Update it at the end of every working session.
- `src/engine/`: pure, deterministic game logic. No DOM, no timers, no network, no `Math.random` (use the seeded RNG).
- `src/encoders/`: state-to-text encoders. They report facts and never verdicts (see RFC decision 5).
- `src/agent/`: agent loop, policies, and the `/v1/systemone` client.
- `src/agent/oracle.ts`: engine lookahead (rollouts) used to label training data.
- `training/`: Python fine-tuning and export scripts (uv `.venv`), run on the GPU machine.
- `src/render/`, `src/main.ts`, `src/audio/`: browser-only code. Read engine state and never mutate game rules.
- `scripts/`: headless runner, probe, and dev/build entrypoints. Scripts are the command contract; do not leave commands only in the README.
- `tests/`: Vitest tests.
- `ios/`: the iPhone app (xcodegen `project.yml`, Swift sources, build and Mac-check scripts). Models, keys, the llama.cpp xcframework, and the web build are not committed.

## Environment

- Node with npm. Vite + TypeScript, Vitest for tests, `tsx` for scripts.
- Ollama 0.35+ running locally with a decision model pulled (`ollama pull tev1:4b`). Default base URL `http://localhost:11434`, overridable with `VITE_DECISION_BASE_URL`.
- The teacher is any OpenAI-compatible chat endpoint set through `TEACHER_BASE_URL` and `TEACHER_MODEL`. Never commit real hostnames for it.
- Python (probes, training) uses a project-local `.venv` created with `uv`. The probes are stdlib-only.

## Rules

- The simulation must never await the model. Keep the 30 Hz fixed step independent of the agent loop.
- Benchmark and latency claims must come from fresh states. Exact repeated requests hit a cache and are 5–10x faster.
- Lookahead (engine rollouts) may label training data but never goes into a player's input in a comparison. Players are compared on the same current-state input.
- Record measured numbers (latency, tokens, scores) in `working.md` with the model, encoder, and hardware.
- Public repo hygiene: no real emails, keys, internal paths, hostnames, or vault references in any tracked file. Use `.env.example` with placeholder values. Run a privacy scan (`rg -n -i "@|op://|/Users/|ts\.net|tailscale|api[_-]?key|token" .` and review every hit) before any push. None of this belongs in the README.
- No original arcade art, audio, or branding. Draw graphics in code.
- Evaluation seeds never produce training data. Keep seed ranges for train, validation, and evaluation disjoint.
- Version control: default branch is `master`, which is protected, so changes land through PRs. Commit, push, or open PRs only when the user explicitly asks. Keep commits small and scoped.
