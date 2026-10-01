# Distill a 0.8B player

The weights of `pacman-0.8b` and `pacman-0.8b-qwen` are not published, and the training data is not in the repository. This guide describes how to generate the data, train both models, and import them into Ollama.

Training requires a CUDA GPU (the published runs used one RTX 5090). Generating the Qwen labels also requires an OpenAI-compatible chat endpoint serving a capable model (the published runs used Qwen3.8-27B NVFP4).

## Overview

Label states → build training rows → train a LoRA → export GGUF → import into Ollama → score.

A player drives the game in lockstep. At every decision, the data generator stores the state in the encoding the student will read (`features`) together with a label from the labeler. The labeler may use lookahead: `oracle-5s` rolls the engine forward, and `teacher-peek5s` shows the chat model one simulated 5-second future per option. The student never gets lookahead in its input. Lookahead may label data but never enters a player's input in a comparison.

## 1. Label states

Data generation runs the game in lockstep using `scripts/gen_teacher_data.ts` (`npm run gen-data -- ...`).

### Oracle labels (v1)

The default labeler is `oracle-5s`. Run three players across the training seeds into `data/v1/train`:

```bash
npm run gen-data -- --player oracle-5s --games 20 --seed 1000 --out data/v1/train
npm run gen-data -- --player greedy --games 30 --seed 2000 --out data/v1/train
npm run gen-data -- --player random --games 30 --seed 3000 --out data/v1/train
```

Run the validation seeds into `data/v1/val`:

```bash
npm run gen-data -- --player oracle-5s --games 3 --seed 900 --out data/v1/val
npm run gen-data -- --player greedy --games 5 --seed 920 --out data/v1/val
npm run gen-data -- --player random --games 5 --seed 940 --out data/v1/val
```

Each game runs for 180 seconds of game time (the default). The v1 runs produce 57,249 training states and 8,050 validation states, which deduplicate to 31,585 training rows and 4,975 validation rows.

### Qwen labels (q1)

Configure `TEACHER_BASE_URL` and `TEACHER_MODEL` in `.env`. Run the same seeds with `--labeler teacher-peek5s`:

```bash
npm run gen-data -- --player oracle-5s --labeler teacher-peek5s --games 20 --seed 1000 --out data/q1/train
npm run gen-data -- --player greedy --labeler teacher-peek5s --games 30 --seed 2000 --out data/q1/train
npm run gen-data -- --player random --labeler teacher-peek5s --games 30 --seed 3000 --out data/q1/train

npm run gen-data -- --player oracle-5s --labeler teacher-peek5s --games 3 --seed 900 --out data/q1/val
npm run gen-data -- --player greedy --labeler teacher-peek5s --games 5 --seed 920 --out data/q1/val
npm run gen-data -- --player random --labeler teacher-peek5s --games 5 --seed 940 --out data/q1/val
```

Players are deterministic for a given seed, so both datasets visit the exact same states. Because the teacher server returned no logprobs, q1 contains hard labels. Generating the 54,325 teacher requests took about 65 minutes at concurrency 8. After deduplication, `pacman-0.8b-qwen` trained on 31,570 training rows and 4,973 validation rows.

`--also-encode features-peek5s` stores a second, lookahead encoding of each state under `alt`, and `build_sft.py --encoder features-peek5s` builds training rows from it. That is how student B (`pacman-0.8b-qwen-peek`) of section 18 was built. Its input contains lookahead, so its scores are records, not ladder rows. You do not need it for `pacman-0.8b-qwen`.

## 2. Build training rows

`training/build_sft.py` packages the states into training rows. The input directory passed to `--data` must contain `train/` and `val/` subdirectories containing the run folders.

For oracle labels (v1), build soft targets from oracle values using the default temperature of 50:

```bash
python training/build_sft.py --data data/v1 --out data/sft
```

For chat teacher labels (q1), build targets from the labels with label smoothing:

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
CUDA_VISIBLE_DEVICES=0 python training/train.py --data data/sft --out runs/ft_v1 --epochs 2
```

For the Qwen-distilled dataset:

```bash
CUDA_VISIBLE_DEVICES=0 python training/train.py --data data/sft_q1 --out runs/ft_q1 --epochs 2
```

The script defaults to `--base Qwen/Qwen3.5-0.8B`, `--batch 32`, `--lr 1e-4`, `--rank 16`, `--alpha 32`, and `--epochs 1.0`. The published models were trained with `--epochs 2`.

On one RTX 5090, training takes about 32-40 minutes for about 1,975 steps. The script writes the LoRA adapter to `<out>/adapter`.

## 4. Export to GGUF

Use `training/export_gguf.sh` to merge the adapter and convert the model to Q8_0 GGUF format:

```bash
training/export_gguf.sh runs/ft_v1/adapter runs/ft_v1/pacman-0.8b-q8_0.gguf /path/to/llama.cpp
training/export_gguf.sh runs/ft_q1/adapter runs/ft_q1/pacman-0.8b-qwen-q8_0.gguf /path/to/llama.cpp
```

The script merges the adapter in the current (training) environment, writes the merged weights next to the adapter (for example `runs/ft_v1/merged`), and converts them with the Python environment at `<llama.cpp_dir>/.venv`. Create that environment from llama.cpp's converter requirements, then upgrade `transformers` in it: the requirements pin 4.57.6, but Qwen3.5 tokenizers need 5.5 or later.

## 5. Import into Ollama

Create the model through Ollama's HTTP API (`/api/create`) rather than a Modelfile, so the system prompt is exactly the one the model was trained under. Ollama 0.35.0 rejects `CAPABILITY` in a Modelfile, but it serves `/v1/systemone` for the imported model anyway.

The commands below import `pacman-0.8b-qwen`. For `pacman-0.8b`, change `model` and `gguf`. The `create.json` body takes the system prompt from the `SYSTEM` string in `training/prompt.py`, so it matches byte for byte, including the leading and trailing newlines.

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

With `model=pacman-0.8b`, this is the request body that created the original `pacman-0.8b`. Check the result with `python3 scripts/probe_decision_models.py pacman-0.8b-qwen`.

`training/prompt.py` formats prompts identically to Ollama's `/v1/systemone` endpoint. On 50 validation states, the served Q8_0 model and the bf16 training weights agreed to within 0.027 in probability (median 0.010).

## 6. Score

### Offline agreement

Measure decision agreement on 1,000 held-out validation states using `training/score_ollama.py`:

```bash
# Agreement with the oracle (data/v1/val), as reported in results.md
python3 training/score_ollama.py --model pacman-0.8b --n 1000
python3 training/score_ollama.py --model pacman-0.8b-qwen --n 1000

# Agreement with the teacher's own labels (data/q1/val)
python3 training/score_ollama.py --model pacman-0.8b-qwen --data data/q1/val --target label --n 1000
```

### In-game evaluation

Run the ladder protocol (10 games, seeds 100-109, 5-minute cap per game, realtime clock):

```bash
npm run headless -- --policy pacman-0.8b --games 10 --seed 100 --max-seconds 300
npm run headless -- --policy pacman-0.8b-qwen --games 10 --seed 100 --max-seconds 300
```

### Parity between training and serving

`training/score_hf.py` scores the same validation states with the merged bf16 weights, so you can compare them with the served model's probabilities:

```bash
python training/score_hf.py --model runs/ft_v1/merged --n 50 --out runs/ft_v1/score_hf.json
```

### Numbers to compare with

From the runs in sections 11 and 18 of [model_evaluation.md](../model_evaluation.md). Agreement is with the oracle on the same 1,000 validation states for both models; `pacman-0.8b-qwen` agreed with its teacher's labels on 61.9%.

| Model | Mean pellets | Runs | Decision p50 | Agreement with the oracle |
|---|---|---|---|---|
| `pacman-0.8b` | 386 | 456, 300, 400 | 59-63 ms | 53.9% |
| `pacman-0.8b-qwen` | 425 | 441, 467, 366 | 55 ms | 55.3% |

Realtime scores for the same model varied by up to 150 pellets between 10-game runs. The 2026-10-01 runs shared the Mac with heavy unrelated CPU load (load average 15-36); `pacman-0.8b` and `pacman-0.8b-qwen` were alternated to face the same operating conditions. Compare realtime scores within one day's runs, not across days. The full comparison is in [results.md](../results.md).

## Tests

Run the unit tests for `build_sft`:

```bash
cd training && python3 -m unittest test_build_sft
```
