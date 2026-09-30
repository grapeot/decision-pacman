# Decision Pac-Man

A local decision model plays a Pac-Man-style maze game in the browser in real time. The game runs at a fixed 30 ticks per second. Each decision is a typed choice among the legal moves, returned with a probability for every option by a model served on your own machine through the `/v1/systemone` API (Ollama 0.35+ or ollaya).

The project goes in four stages:

1. A real-time browser game steered by `tev1:4b`.
2. Teacher data from a large chat model playing and labeling positions in the same engine.
3. Fine-tuning a 0.8B decision model on that data.
4. Evaluating the fine-tuned model against off-the-shelf models on held-out games.

**Status:** design stage. The plan is in [`docs/prd.md`](docs/prd.md) and [`docs/rfc.md`](docs/rfc.md). The model measurements behind it are in [`docs/model_evaluation.md`](docs/model_evaluation.md).

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
