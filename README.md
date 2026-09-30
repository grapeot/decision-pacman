# Decision Pac-Man

A local decision model plays a Pac-Man-style maze game in the browser in real time. The game runs at a fixed 30 ticks per second. Each decision is a typed choice among the legal moves, returned with a probability for every option by a model served on your own machine through the `/v1/systemone` API (Ollama 0.35+ or ollaya).

The project goes in four stages:

1. A real-time browser game steered by `tev1:4b`.
2. Teacher data from a large chat model playing and labeling positions in the same engine.
3. Fine-tuning a 0.8B decision model on that data.
4. Evaluating the fine-tuned model against off-the-shelf models on held-out games.

**Status:** Stage 1 works. The game runs in the browser with `tev1:4b` playing in real time. A 30-second recording is in [`docs/media/demo_tev1_4b.mp4`](docs/media/demo_tev1_4b.mp4). The plan is in [`docs/prd.md`](docs/prd.md) and [`docs/rfc.md`](docs/rfc.md). The measurements are in [`docs/model_evaluation.md`](docs/model_evaluation.md).

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

A chat model on any OpenAI-compatible endpoint can play or label instead (`--player teacher --labeler teacher`, with `TEACHER_BASE_URL` and `TEACHER_MODEL` in `.env`).

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
