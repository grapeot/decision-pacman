# Decision Pac-Man

A model steers a Pac-Man-style game in the browser in real time at 30 ticks per second, choosing among the legal moves at each junction. The repository contains the browser game, a headless runner for seeded games, a search oracle, data generation scripts, fine-tuning scripts for a 0.8B model, evaluation benchmarks, and an iPhone app.

## Results

Every player sees the same current-state input (`features`), with no lookahead. Each row is 10 games on evaluation seeds 100-109, at 1x speed with a 5-minute cap, on the realtime clock: a slow answer costs game time, as it does in the browser.

| Player | Mean pellets | Decision p50 |
|---|---|---|
| random | 30 | 0 ms |
| `tev1:0.8b` | 69 | 62 ms |
| `tev1:4b` | 148 | 202 ms |
| Jev 1.13.0 (hosted) | 178 | 115 ms |
| `llm:qwen3.5:4b` | 197 | 267 ms |
| greedy (scripted) | 201 | 0 ms |
| `llm:gemma4:e4b` | 230 | 253 ms |
| `llm:phi4-mini` | 232 | 187 ms |
| `pacman-0.8b` (fine-tuned on `oracle-5s` labels) | 386, mean of 3 runs (456, 300, 400) | 59-63 ms |
| `pacman-0.8b-qwen` (fine-tuned on Qwen3.8-27B labels; features input) | 425, mean of 3 runs (441, 467, 366) | 55 ms |

Lockstep results are listed apart, because there the game waits for each answer and the numbers are not comparable with the ladder: Qwen3.8-27B scored 301 pellets with thinking off and 531 with thinking on (decision p50 2.4 s), both over 3 games on seeds 100-102; `pacman-0.8b-qwen` scored 417 over seeds 100-109.

Full numbers, caveats (clock effects, run-to-run variance up to ~150 pellets, hardware), and the lookahead rule are in [docs/results.md](docs/results.md).

## What we found

- A 0.8B model (LoRA on the Qwen3.5-0.8B base) fine-tuned on 31,585 oracle-labeled states averaged 386 pellets (3-run mean) at about 60 ms per decision, compared to 69 for the same-size off-the-shelf `tev1:0.8b`.
- Distilling Qwen3.8-27B's lookahead labels into a 0.8B model that reads only `features` gave 425 pellets (3-run mean), level with the oracle-trained student given run-to-run variance. A general LLM with lookahead can replace the hand-built search as the teacher.
- Stock small chat models with constrained JSON (`llm:phi4-mini` 232, `llm:gemma4:e4b` 230, `llm:qwen3.5:4b` 197) matched or beat the decision models `tev1:4b` (148) and Jev (178) from the same facts.
- Specialization cost generality: on 194 JevBench items, `pacman-0.8b` scored 0.49 accuracy against 0.66 for `tev1:0.8b` (chance 0.32).
- Decisions are prefill-bound (`docs/model_evaluation.md` section 17): decision models read the answer directly from option logits, whereas chat models writing ~6-token JSON pay about 72 ms more on an Apple M3 Ultra and 27 ms on an RTX 5090.

## Watch

30-second clips, all on seed 100 at 1x speed:

| Clip | Player | Pellets | Lives lost |
|---|---|---|---|
| [`docs/media/demo_tev1_08b.mp4`](docs/media/demo_tev1_08b.mp4) | `tev1:0.8b` | 71 | 2 |
| [`docs/media/demo_tev1_4b.mp4`](docs/media/demo_tev1_4b.mp4) | `tev1:4b` | 101 | 1 |
| [`docs/media/demo_jev.mp4`](docs/media/demo_jev.mp4) | Jev 1.13.0 (hosted) | 86 | 1 |
| [`docs/media/demo_qwen35_4b.mp4`](docs/media/demo_qwen35_4b.mp4) | `llm:qwen3.5:4b` | 109 | 0 |
| [`docs/media/demo_pacman_08b.mp4`](docs/media/demo_pacman_08b.mp4) | `pacman-0.8b` | 170 | 0 |

## Quickstart

Requirements: Node 20+, Ollama 0.35+.

```bash
ollama pull tev1:4b
npm install
npm run dev            # open http://localhost:5173
```

Switch models, encoders, and speed in the side panel, or pass URL parameters such as `?model=tev1:4b`.

## iPhone

`ios/` runs the same game on an iPhone with three players: the distilled 0.8B and phi4-mini on device, and Jev in the cloud. See [docs/guides/iphone.md](docs/guides/iphone.md).

## Reproduce each result

| Result | Command | Guide |
|---|---|---|
| Random and greedy baselines | `npm run headless -- --policy random --games 10 --seed 100 --max-seconds 300`<br>`npm run headless -- --policy greedy --games 10 --seed 100 --max-seconds 300` | [docs/guides/evaluate.md](docs/guides/evaluate.md), [docs/model_evaluation.md](docs/model_evaluation.md) section 11 |
| `tev1:0.8b` and `tev1:4b` | `npm run headless -- --policy tev1:0.8b --games 10 --seed 100 --max-seconds 300`<br>`npm run headless -- --policy tev1:4b --games 10 --seed 100 --max-seconds 300` | [docs/guides/run-models.md](docs/guides/run-models.md), [docs/guides/evaluate.md](docs/guides/evaluate.md), [docs/model_evaluation.md](docs/model_evaluation.md) section 11 |
| Jev (hosted) | `npm run headless -- --policy jev --games 10 --seed 100 --max-seconds 300`<br>(with `TYPESAFE_API_KEY` set; take the key from your password manager) | [docs/guides/run-models.md](docs/guides/run-models.md), [docs/guides/evaluate.md](docs/guides/evaluate.md), [docs/model_evaluation.md](docs/model_evaluation.md) sections 11, 12 |
| Plain chat models (`llm:phi4-mini` etc.) | `ollama pull phi4-mini`<br>`npm run headless -- --policy llm:phi4-mini --games 10 --seed 100 --max-seconds 300` | [docs/guides/run-models.md](docs/guides/run-models.md), [docs/guides/evaluate.md](docs/guides/evaluate.md), [docs/model_evaluation.md](docs/model_evaluation.md) section 14 |
| `pacman-0.8b` (train it yourself; weights not published) | `npm run headless -- --policy pacman-0.8b --games 10 --seed 100 --max-seconds 300` | [docs/guides/distill.md](docs/guides/distill.md), [docs/guides/evaluate.md](docs/guides/evaluate.md), [docs/model_evaluation.md](docs/model_evaluation.md) sections 11, 18 |
| `pacman-0.8b-qwen` (train it yourself; weights not published) | `npm run headless -- --policy pacman-0.8b-qwen --games 10 --seed 100 --max-seconds 300` | [docs/guides/distill.md](docs/guides/distill.md), [docs/guides/evaluate.md](docs/guides/evaluate.md), [docs/model_evaluation.md](docs/model_evaluation.md) section 18 |
| Lockstep rows | `npm run headless -- --policy teacher --clock lockstep --games 3 --seed 100 --max-seconds 300`<br>`npm run headless -- --policy teacher-think --clock lockstep --games 3 --seed 100 --max-seconds 300`<br>`npm run headless -- --policy pacman-0.8b-qwen --clock lockstep --games 10 --seed 100 --max-seconds 300`<br>(the teacher needs `TEACHER_BASE_URL` and `TEACHER_MODEL` in `.env`) | [docs/guides/evaluate.md](docs/guides/evaluate.md), [docs/model_evaluation.md](docs/model_evaluation.md) sections 10, 13, 18 |
| Agreement with the oracle | `python3 training/score_ollama.py --model <name> --n 1000` | [docs/guides/evaluate.md](docs/guides/evaluate.md), [docs/model_evaluation.md](docs/model_evaluation.md) sections 11, 14, 18 |
| Generality | `python3 scripts/eval_general_decisions.py tev1:4b tev1:0.8b` | [docs/guides/evaluate.md](docs/guides/evaluate.md), [docs/model_evaluation.md](docs/model_evaluation.md) sections 15, 18 |
| Latency breakdown | `python3 scripts/bench_prefill_decode.py --backend ollama --model tev1:4b` | [docs/guides/evaluate.md](docs/guides/evaluate.md), [docs/model_evaluation.md](docs/model_evaluation.md) section 17 |
| Clips | `npm run record -- --model tev1:4b --seconds 30 --seed 100 --out docs/media/demo.mp4` | [docs/guides/run-models.md](docs/guides/run-models.md) |

## Guides

- [docs/guides/run-models.md](docs/guides/run-models.md): Play any model in the browser or headless runner.
- [docs/guides/evaluate.md](docs/guides/evaluate.md): Evaluation protocol, seeds, clocks, variance, and scoring scripts.
- [docs/guides/distill.md](docs/guides/distill.md): Label, train, export, import, and score a fine-tuned model (requires a CUDA GPU).
- [docs/guides/iphone.md](docs/guides/iphone.md): Build and run the native iOS app with three selectable players.
- [docs/results.md](docs/results.md): Canonical current numbers, hardware details, caveats, and the lookahead rule.
- [docs/model_evaluation.md](docs/model_evaluation.md): Chronological lab notebook across experiments 1 through 18.
- [docs/rfc.md](docs/rfc.md): Architecture and technical decisions.
- [docs/prd.md](docs/prd.md): Original requirements.
- [docs/working.md](docs/working.md): Changelog and development notes.

## Tests

```bash
npm test
```

## License

MIT. Pac-Man is a trademark of Bandai Namco. This project is not affiliated with it and uses no original art or audio.
