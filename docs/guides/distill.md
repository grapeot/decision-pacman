# Distill a 0.8B player

The distilled `pacman-0.8b-qwen` is published on Hugging Face at [grapeot/decision-pacman-0.8b-GGUF](https://huggingface.co/grapeot/decision-pacman-0.8b-GGUF). To play it, you need no GPU: download and import it as in [run-models.md](run-models.md). This guide rebuilds it from scratch: label states with an LLM teacher, train, export, import into Ollama, and score. The training data is not in the repository.

Training requires a CUDA GPU (the published run used one RTX 5090), or an Apple-silicon Mac with MLX: [distill-mlx.md](distill-mlx.md) replaces steps 3 and 4 below. Labeling requires an OpenAI-compatible chat endpoint serving a capable model (the published run used Qwen3.8-27B NVFP4); a hosted API serving the same model works too, see [teacher-api.md](teacher-api.md).

## Overview

Label states → build training rows → train a LoRA → export GGUF → import into Ollama → score.

A player drives the game in lockstep. At every decision, the data generator stores the state in the encoding the student will read (`features`) together with the teacher's choice. The teacher (`teacher-peek5s`) is the chat model shown one simulated 5-second future per option, computed by the engine rollouts in `src/agent/oracle.ts`. That lookahead is used only to make labels. The student never gets it in its input: lookahead may label data but never enters a player's input in a comparison.

## 1. Label states

Data generation runs the game in lockstep using `scripts/gen_teacher_data.ts` (`npm run gen-data -- ...`).

Set `TEACHER_BASE_URL` and `TEACHER_MODEL` in `.env` (see `.env.example`). To label without a GPU, point them at a hosted API as described in [teacher-api.md](teacher-api.md); hosted APIs also need `TEACHER_API_KEY` and a switch that turns thinking off. Run three players across the training seeds into `data/q1/train`, and across the validation seeds into `data/q1/val`, all labeled by the teacher:

```bash
npm run gen-data -- --player oracle-5s --labeler teacher-peek5s --games 20 --seed 1000 --out data/q1/train
npm run gen-data -- --player greedy --labeler teacher-peek5s --games 30 --seed 2000 --out data/q1/train
npm run gen-data -- --player random --labeler teacher-peek5s --games 30 --seed 3000 --out data/q1/train

npm run gen-data -- --player oracle-5s --labeler teacher-peek5s --games 3 --seed 900 --out data/q1/val
npm run gen-data -- --player greedy --labeler teacher-peek5s --games 5 --seed 920 --out data/q1/val
npm run gen-data -- --player random --labeler teacher-peek5s --games 5 --seed 940 --out data/q1/val
```

The player only decides which states get visited; the teacher labels every one of them. The mix covers positions a strong player reaches (`oracle-5s`, a search that plays from the same rollouts), positions a simple rule reaches (`greedy`), and poor positions (`random`). Each game runs for 180 seconds of game time (the default). Players are deterministic for a given seed, so a rerun visits the same states.

The teacher server returned no logprobs, so the labels are hard. Generating the 54,325 teacher requests took about 65 minutes at about 14 requests per second. After deduplication, `pacman-0.8b-qwen` trained on 31,570 training rows and 4,973 validation rows.

`--also-encode features-peek5s` stores a second, lookahead encoding of each state under `alt`, and `build_sft.py --encoder features-peek5s` builds training rows from it. That is how student B (`pacman-0.8b-qwen-peek`) of section 18 of [model_evaluation.md](../model_evaluation.md) was built. Its input contains lookahead, so its scores are records, not ladder rows. You do not need it for `pacman-0.8b-qwen`.

Note: without `--labeler`, `gen-data` labels each state with the `oracle-5s` search itself and stores every option's rollout value. The earlier `pacman-0.8b` was trained that way (`python training/build_sft.py --data data/v1 --out data/sft`, soft targets at temperature 50) and played at the same level; see the FAQ in the [README](../../README.md#faq).

## 2. Build training rows

`training/build_sft.py` packages the states into training rows. The directory passed to `--data` must contain `train/` and `val/` subdirectories holding the run folders. Build targets from the teacher's labels, with 0.1 of the probability spread over the other options:

```bash
python training/build_sft.py --data data/q1 --out data/sft_q1 --target label --smoothing 0.1
```

## 3. Train

Create a uv environment and install the dependencies listed in `training/pyproject.toml` (`torch>=2.8`, `transformers>=5.5`, `peft>=0.17`, `unsloth>=2026.9`):

```bash
uv venv training/.venv
source training/.venv/bin/activate
uv pip install -r training/pyproject.toml
```

Run training on a CUDA GPU:

```bash
CUDA_VISIBLE_DEVICES=0 python training/train.py --data data/sft_q1 --out runs/ft_q1 --epochs 2
```

The script defaults to `--base Qwen/Qwen3.5-0.8B`, `--batch 32`, `--lr 1e-4`, `--rank 16`, `--alpha 32`, and `--epochs 1.0`. The published model was trained with `--epochs 2`.

On one RTX 5090, training takes about 32 minutes for 1,974 steps. The script writes the LoRA adapter to `<out>/adapter`. To train on a Mac instead, see [distill-mlx.md](distill-mlx.md).

## 4. Export to GGUF

Use `training/export_gguf.sh` to merge the adapter and convert the model to Q8_0 GGUF format:

```bash
training/export_gguf.sh runs/ft_q1/adapter runs/ft_q1/pacman-0.8b-qwen-q8_0.gguf /path/to/llama.cpp
```

The script merges the adapter in the current (training) environment, writes the merged weights next to the adapter (`runs/ft_q1/merged`), and converts them with the Python environment at `<llama.cpp_dir>/.venv`. Create that environment from llama.cpp's converter requirements, then upgrade `transformers` in it: the requirements pin 4.57.6, but Qwen3.5 tokenizers need 5.5 or later.

## 5. Import into Ollama

The model must be served with exactly the system prompt it was trained under: the `SYSTEM` string in `training/prompt.py`, including the leading and trailing newlines. Ollama 0.35.0 rejects `CAPABILITY` in a Modelfile, but it serves `/v1/systemone` for the imported model anyway.

The commands below create the model through Ollama's HTTP API (`/api/create`), taking the system prompt straight from `training/prompt.py`:

```bash
model=pacman-0.8b-qwen
gguf=runs/ft_q1/pacman-0.8b-qwen-q8_0.gguf
digest="$(shasum -a 256 "$gguf" | cut -d' ' -f1)"
curl -T "$gguf" -X POST "http://localhost:11434/api/blobs/sha256:$digest"

python3 - "$model" "$digest" > create.json <<'EOF'
import json, sys
sys.path.insert(0, "training")
from prompt import SYSTEM
model, digest = sys.argv[1], sys.argv[2]
print(json.dumps({"model": model, "files": {model + "-q8_0.gguf": "sha256:" + digest},
                  "system": SYSTEM, "parameters": {"num_ctx": 2050},
                  "capabilities": ["decision"], "stream": False}))
EOF

curl http://localhost:11434/api/create -d @create.json
```

The published model takes the other route: its Modelfile on Hugging Face (`Modelfile.pacman-0.8b-qwen`) names the GGUF in `FROM`, sets `TEMPLATE {{ .Prompt }}`, and carries the same system prompt, for `ollama create pacman-0.8b-qwen -f Modelfile.pacman-0.8b-qwen`. Imported that way, it gave the same choices and probabilities as the evaluated model on 100 validation states.

Check the result with `python3 scripts/probe_decision_models.py pacman-0.8b-qwen`.

`training/prompt.py` formats prompts identically to Ollama's `/v1/systemone` endpoint. On 50 validation states, the served Q8_0 model and the bf16 training weights chose the same option on 50 of 50.

## 6. Score

### Offline agreement

Measure how often the model picks the teacher's move on 1,000 held-out validation states using `training/score_ollama.py`:

```bash
python3 training/score_ollama.py --model pacman-0.8b-qwen --data data/q1/val --target label --n 1000
```

`results.md` also reports agreement with the `oracle-5s` search's best option. That needs validation states labeled by the search (the `gen-data` default labeler) in `data/v1/val`, the script's default `--data`.

### In-game evaluation

Run the ladder protocol (10 games, seeds 100-109, 5-minute cap per game, realtime clock):

```bash
npm run headless -- --policy pacman-0.8b-qwen --games 10 --seed 100 --max-seconds 300
```

### Parity between training and serving

`training/score_hf.py` scores the same validation states with the merged bf16 weights, so you can compare them with the served model's probabilities:

```bash
python training/score_hf.py --model runs/ft_q1/merged --data data/q1/val --target label --n 50 --out runs/ft_q1/score_hf.json
```

### Numbers to compare with

From section 18 of [model_evaluation.md](../model_evaluation.md):

| Model | Mean pellets (realtime) | Runs | Decision p50 | Lockstep | Agreement with the teacher |
|---|---|---|---|---|---|
| `pacman-0.8b-qwen` | 425 | 441, 467, 366 | 55 ms | 417 | 61.9% |

Realtime scores for the same model varied by up to 150 pellets between 10-game runs. The 2026-10-01 runs shared the Mac with heavy unrelated CPU load (load average 15-36), so the models compared that day were alternated to face the same conditions. Compare realtime scores within one day's runs, not across days. The full comparison is in [results.md](../results.md).

## Tests

Run the unit tests for `build_sft`:

```bash
cd training && python3 -m unittest test_build_sft
```
