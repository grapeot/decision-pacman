# Decision Pac-Man

A local decision model plays a Pac-Man-style maze game in the browser in real time. The game runs at a fixed 30 ticks per second. Each decision is a typed choice among the legal moves, returned with a probability for every option by a model served on your own machine through the `/v1/systemone` API (Ollama 0.35+ or ollaya).

The project goes in four stages:

1. A real-time browser game steered by `tev1:4b`.
2. Teacher data from a large chat model playing and labeling positions in the same engine.
3. Fine-tuning a 0.8B decision model on that data.
4. Evaluating the fine-tuned model against off-the-shelf models on held-out games.

**Status:** Stages 1 to 3 work. A 0.8B model fine-tuned on search-oracle labels averages 456 pellets per game at 63 ms per decision, against 69 for the same-size off-the-shelf `tev1:0.8b` and 148 for `tev1:4b`. The plan is in [`docs/prd.md`](docs/prd.md) and [`docs/rfc.md`](docs/rfc.md). The measurements are in [`docs/model_evaluation.md`](docs/model_evaluation.md).

## Three models, same game

Each clip is 30 seconds on the same seed at 1x speed:

| Clip | Model | Score | Pellets | Lives lost |
|---|---|---|---|---|
| [`demo_tev1_08b.mp4`](docs/media/demo_tev1_08b.mp4) | `tev1:0.8b`, off the shelf, about 70 ms | 710 | 71 | 2 |
| [`demo_tev1_4b.mp4`](docs/media/demo_tev1_4b.mp4) | `tev1:4b`, off the shelf, about 200 ms | 1,250 | 101 | 1 |
| [`demo_pacman_08b.mp4`](docs/media/demo_pacman_08b.mp4) | `pacman-0.8b`, fine-tuned, about 60 ms | 2,660 | 170 | 0 |

The small model is fast but its option probabilities are close to uniform. The larger one plays better but answers three times slower. After fine-tuning, the small model plays better than both at the small model's speed.

## Run it

You need Node 20+ and Ollama 0.35+:

```sh
ollama pull tev1:4b
npm install
npm run dev          # open http://localhost:5173
```

The dev server forwards `/decide` to `http://localhost:11434`. Set `VITE_DECISION_BASE_URL` to use another endpoint, such as ollaya on port 11435. In the side panel you can switch models, encoders, and game speed, or take over with the keyboard.

Play without a browser and write per-decision logs to `runs/`:

```sh
npm run headless -- --policy tev1:4b --games 3 --seed 100
npm run headless -- --policy greedy --games 10    # scripted baseline
```

Record a clip (needs Playwright's Chromium and ffmpeg):

```sh
npm run record -- --model tev1:4b --seconds 30 --out docs/media/demo.mp4
```

Generate labeled training states. By default a search oracle plays and labels every decision with a value per option:

```sh
npm run gen-data -- --games 20 --seed 1000
npm run gen-data -- --player random --games 20 --seed 2000   # other players, oracle labels
```

A chat model on any OpenAI-compatible endpoint can play or label instead (`--player teacher --labeler teacher`, with `TEACHER_BASE_URL` and `TEACHER_MODEL` in `.env`). Players are deterministic for a seed, so the same player and seeds with another labeler give the same states with new labels. `--also-encode features-peek5s` also stores each state in that encoding, for a student that reads it:

```sh
npm run gen-data -- --player oracle-5s --labeler teacher-peek5s --also-encode features-peek5s --games 5 --seed 1000
python3 training/build_sft.py --data data/q1 --out data/sft_q1 --target label --smoothing 0.1
python3 training/build_sft.py --data data/q1 --out data/sft_q1_peek --target label --smoothing 0.1 --encoder features-peek5s
```

Fine-tune a model on the labeled states (a CUDA GPU; see `training/pyproject.toml`), export it to GGUF, and score it through Ollama:

```sh
python3 training/build_sft.py --data data/v1 --out data/sft
python training/train.py --data data/sft --out runs/ft_v1 --epochs 2
training/export_gguf.sh runs/ft_v1/adapter runs/ft_v1/model-q8_0.gguf /path/to/llama.cpp
python3 training/score_ollama.py --model your-model --data data/v1/val --n 1000
```

`training/prompt.py` renders prompts exactly as Ollama's `/v1/systemone` does, so the served model sees what it was trained on. Import the GGUF with Ollama's `/api/create` and the same system prompt as `tev1`.

Run the tests with `npm test`.

## Try the probes

The probes need Python 3.9+ (standard library only) and a running decision model:

```sh
ollama pull tev1:4b
python3 scripts/probe_decision_models.py tev1:4b
```

To probe a teacher model over any OpenAI-compatible endpoint:

```sh
TEACHER_BASE_URL=http://localhost:8000/v1 TEACHER_MODEL=your-model \
  python3 scripts/probe_teacher.py nothink
```

## License

MIT. Pac-Man is a trademark of Bandai Namco. This project is not affiliated with it and uses no original art or audio.
