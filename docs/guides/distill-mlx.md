# Distill the 0.8B on a Mac (MLX)

[distill.md](distill.md) trains `pacman-0.8b-qwen` on a CUDA GPU. This guide replaces its training and export steps (3 and 4) with an Apple-silicon path built on [MLX](https://github.com/ml-explore/mlx) and `mlx-lm`, and measures what that costs. Labeling (step 1), building rows (step 2), importing into Ollama (step 5), and scoring (step 6) are unchanged.

The Mac run uses the same 31,570 training and 4,973 validation rows, the same chat-templated prompt, the same loss, and the same hyperparameters as the published model. On one Mac Studio (Apple M3 Ultra, 80-core GPU, 512 GB), one epoch (987 steps) took 75 minutes, at 4.2 seconds per step against about 0.9 on one RTX 5090. The published model trained for two epochs, which on the Mac would take about 2 hours 26 minutes; a two-epoch run measured over its first 1,600 steps showed no gain in validation agreement after step 600. Served through Ollama, it averaged 371 pellets over three realtime 10-game runs and 352 in lockstep, at 53 ms per decision; the published model, alternated with it in the same session, scored 427 and 417. The realtime gap is within the run-to-run spread, but the lockstep gap and a 2.5-point lower agreement with the teacher point the same way, so expect a one-epoch Mac student slightly below the published one. Numbers are in [Measured results](#measured-results) and section 20 of [model_evaluation.md](../model_evaluation.md).

## What you need

- A Mac with Apple silicon. The measured run peaked at 48 GB of GPU memory; smaller settings fit in under 5 GB (see [Memory and speed settings](#memory-and-speed-settings)).
- `uv`, Python 3.12, Node 20+ with npm, and Ollama 0.35+. The measured run used Ollama 0.35.1.
- The training rows from steps 1 and 2 of [distill.md](distill.md), in `data/sft_q1/train.jsonl` and `data/sft_q1/val.jsonl`. The training data is not in the repository, and labeling needs an OpenAI-compatible chat endpoint serving a capable teacher model.
- A llama.cpp checkout recent enough to convert Qwen3.5 (`Qwen3_5ForConditionalGeneration` in `convert_hf_to_gguf.py`), for the GGUF export.

## 1. Environment

```bash
uv venv training/.venv -p 3.12
source training/.venv/bin/activate
uv pip install -r training/requirements-mlx.txt
```

`training/requirements-mlx.txt` pins the versions the measured run used: `mlx` 0.32.3 and `mlx-lm` 0.32.0. `mlx-lm` 0.32 loads Qwen3.5 (`mlx_lm/models/qwen3_5.py`) straight from the Hugging Face checkpoint `Qwen/Qwen3.5-0.8B`, the same base as the CUDA run; the first run downloads it (about 1.7 GB).

## 2. Train

```bash
python training/train_mlx.py --data data/sft_q1 --out runs/ft_q1_mlx --epochs 1 --micro-batch 16
```

This is the measured run. `--epochs 2` is the published model's recipe (1,974 steps) and takes twice as long.

`training/train_mlx.py` is the MLX counterpart of `training/train.py` and keeps its recipe:

- Base `Qwen/Qwen3.5-0.8B` in bf16, LoRA rank 16 on the same 12 linear layer types (attention, Gated DeltaNet, MLP) in all 24 layers, no dropout. PEFT scales the LoRA update by alpha / rank and MLX by `scale`, so alpha 32 becomes scale 2.0 (`--alpha 32` is the default).
- The same loss: the logits of the option letters at the answer position, softmax over those letters only, cross-entropy against the row's 0.9/0.1 target.
- AdamW at lr 1e-4 with no weight decay, 3% linear warmup times a cosine over the run, gradient clipping at norm 1.0, 32 rows per step, and the same shuffle of rows (`random.Random(0)`). The optimizer corrects its moment estimates for their zero start as torch's AdamW does; MLX's AdamW does that only with `bias_correction=True`, and without it the first few hundred updates are several times larger.
- Every 200 steps, the same validation check (cross-entropy and agreement on the first 2,000 validation rows) that `train.py` logs.

`--micro-batch 16` runs each 32-row step as two passes of 16 and averages their gradients, which gives the same update as one pass of 32 because the loss is a mean over rows. It is faster than a single pass of 32 on the M3 Ultra and halves the memory.

At every validation check the script saves the adapter and a checkpoint (optimizer state and position in the data). If the run stops, the same command with `--resume` continues it; on a short test the resumed run reproduced the uninterrupted one's losses exactly. For a run of hours, start it detached from the terminal (`nohup ... &`).

The script writes `runs/ft_q1_mlx/adapter/` (`adapters.safetensors` and `adapter_config.json`, the format `mlx_lm.load(..., adapter_path=...)` reads), `log.jsonl` with the validation checks, and `summary.json` with wall time, tokens per second, and peak memory.

### Why the script swaps in its own Gated DeltaNet recurrence

Qwen3.5 interleaves Gated DeltaNet (linear attention) layers with full attention. `mlx-lm` 0.32 runs those layers with a Metal kernel for inference, but in training mode it falls back to a loop over every token that keeps a `[batch, 16, 128, 128]` float32 state per token for the backward pass. With 32 prompts of about 370 tokens that does not fit in 512 GB. With 8-row micro-batches and gradient checkpointing it fits in 29 GB but takes about 32 seconds per step, about 17.5 hours for the run.

`train_mlx.py` therefore routes the training-mode call to `chunked_gated_delta`, the chunked form of the same recurrence (UT transform within 64-token chunks, a scan across chunks). That is the algorithm Hugging Face transformers uses in its torch fallback, which is what the CUDA run executed. Inference still uses the Metal kernel. `--recurrence stock` restores `mlx-lm`'s loop for comparison.

`training/test_train_mlx.py` checks the chunked recurrence against `mlx-lm`'s loop: outputs and states agree to about 1e-7 and gradients for every input to about 1e-5. On real prompts, the letter logits from the Metal kernel, `mlx-lm`'s loop, and the chunked form differ by at most 0.15 (logits of size about 25, bf16 noise) and pick the same option.

```bash
cd training && python -m unittest test_train_mlx
```

### Memory and speed settings

Measured on the M3 Ultra over the first 3 steps, without the cache limit described below; the full run with `--micro-batch 16` and the default cache limit averaged 2,807 tokens per second (prompt tokens, without padding):

| `--micro-batch` | `--grad-checkpoint` | Peak GPU memory | Tokens per second |
|---|---|---|---|
| 32 | no | 85 GB | 2,600 |
| 16 | no | 43 GB | 3,100 |
| 8 | no | 23 GB | 2,950 |
| 4 | no | 13 GB | 2,800 |
| 8 | yes | 4.6 GB | 2,350 |
| 4 | yes | 3.7 GB | 2,170 |
| 1 | yes | 2.5 GB | 1,590 |

The update is the same in every row; only memory and speed change. For a Mac with 16 GB, start with `--micro-batch 8 --grad-checkpoint`.

`--cache-limit-gb` (default 4) caps MLX's buffer cache. Batches pad to different lengths, so freed buffers are rarely reused, and without a cap the cache grew by about 50 GB per step until macOS killed the process.

### Estimating the time on another Mac

Only the M3 Ultra was measured. Training here is limited by GPU compute, so a Mac with fewer GPU cores will be slower, by more than the core count alone if it also has to use gradient checkpointing. To estimate, time a short run with the settings that fit your memory and multiply:

```bash
python training/train_mlx.py --data data/sft_q1 --out runs/time_check --epochs 1 --micro-batch 8 --grad-checkpoint --max-steps 40 --eval-every 100000
```

`--max-steps` stops early but keeps the full run's learning-rate schedule. Skip the first few steps, which include compilation. A one-epoch run takes about 987 times the seconds per step in the log, plus 5 validation checks (each takes about as long as the "before training" check in the log).

## 3. Export to GGUF

`training/export_gguf_mlx.sh` folds the adapter into the base checkpoint and converts it to Q8_0, like `export_gguf.sh` does for the CUDA path:

```bash
training/export_gguf_mlx.sh runs/ft_q1_mlx/adapter runs/ft_q1_mlx/pacman-0.8b-qwen-mlx-q8_0.gguf /path/to/llama.cpp
```

`training/merge_mlx.py` writes `runs/ft_q1_mlx/merged` in the base model's own Hugging Face layout (same files and tensor names, bf16), with each adapted weight replaced by W + 2.0 · (A·B)ᵀ. The converter runs in the Python environment at `<llama.cpp_dir>/.venv`; create it from llama.cpp's `requirements/requirements-convert_hf_to_gguf.txt`, then upgrade `transformers` to 5.5 or later as [distill.md](distill.md) explains. The merge and conversion took 14 seconds. The Q8_0 file is 795 MB, the same size as the published one.

## 4. Import into Ollama

Follow step 5 of [distill.md](distill.md) with a different model name and the new file, so the published model stays installed for comparison:

```bash
model=pacman-0.8b-qwen-mlx
gguf=runs/ft_q1_mlx/pacman-0.8b-qwen-mlx-q8_0.gguf
```

That route passes `"capabilities": ["decision"]` to `/api/create`. Ollama 0.35.1 serves `/v1/systemone` only for models that declare the decision capability and answers others with "does not support decision"; on 0.35.1 the API route sets it. If you import with a Modelfile instead, add the line `CAPABILITY decision`.

## 5. Score

Check that Ollama serves what was trained. `training/score_mlx.py` is the MLX twin of `score_hf.py`; it scores the same 50 validation states as `score_ollama.py` and writes the same format:

```bash
python training/score_mlx.py --model runs/ft_q1_mlx/merged --data data/q1/val --target label --n 50 --out runs/ft_q1_mlx/score_mlx.json
python3 training/score_ollama.py --model pacman-0.8b-qwen-mlx --data data/q1/val --target label --n 50 --out runs/ft_q1_mlx/parity_ollama.json
```

Then run the agreement check and the in-game protocol of step 6 of [distill.md](distill.md) with `pacman-0.8b-qwen-mlx` in place of `pacman-0.8b-qwen`. Realtime scores vary by up to about 150 pellets between 10-game runs, so alternate the two models in one session and compare means.

## Measured results

Apple M3 Ultra (80-core GPU, 512 GB), macOS 27.0, `mlx` 0.32.3, `mlx-lm` 0.32.0, Ollama 0.35.1. The published model is `pacman-0.8b-qwen` from Hugging Face, alternated with the Mac student in the same session.

| | Mac student, 1 epoch | Published, RTX 5090, 2 epochs |
|---|---|---|
| Training steps | 987 | 1,974 |
| Seconds per step | 4.22 | 0.87 |
| Training loop, including validation checks | 74.5 min | 31 min |
| Prompt tokens per second | 2,807 | |
| Peak GPU memory | 47.8 GB (`--micro-batch 16`) | |
| Validation agreement at the end of training (2,000 rows) | 59.6% | 61.9% |
| Same choice as the training weights, served Q8_0 (50 states) | 49 of 50 | 50 of 50 |
| Agreement with the teacher, served (1,000 states) | 59.4% | 61.9% |
| Realtime pellets, 3 runs of 10 games | 343, 413, 359 (mean 371) | 366, 403, 511 (mean 427) |
| Lockstep pellets, 10 games | 352 | 417 |
| Decision p50 | 53 ms | 53 ms |

A two-epoch Mac run ran 1,600 of its 1,974 steps at 4.13 s per step before it was stopped, which puts two epochs at about 2 hours 26 minutes. Its validation agreement was 59.6% at step 600 and 59.3% at step 1,400. Section 19 of [model_evaluation.md](../model_evaluation.md) has the full curves.

## Differences from the CUDA run

- The measured run trained for one epoch (987 steps), the published model for two (1,974). Everything else in the recipe is the same.
- The Gated DeltaNet recurrence in training is the chunked form described above, not `mlx-lm`'s per-token loop. The math is the same; float32 rounding differs.
- Each 32-row step runs as two micro-batches of 16 with averaged gradients.
- LoRA starts from MLX's own random initialization (the CUDA run seeded PEFT with 3407), so the runs are not bit-identical.
- `train.py` loads the model through Unsloth with gradient checkpointing on; `train_mlx.py` keeps all activations unless `--grad-checkpoint` is set. This changes memory and speed, not the update.

## Caveats

- In `mlx` 0.32.3 the gradient of a step-2 slice (`x[0::2]`) over an axis of length 2 is wrong: it also adds the gradient into the other element. The forward value is right, so the error is silent. `unit_lower_inverse` in `train_mlx.py` avoids such slices, and `test_train_mlx.py` checks its gradient against the exact one.
- `train_mlx.py` replaces `gated_delta_update` in `mlx_lm.models.qwen3_5` and relies on its call signature in `mlx-lm` 0.32.0. Rerun `test_train_mlx.py` after upgrading `mlx-lm`.
- The Mac was otherwise idle during the measured one-epoch run (load average 3-6). The two-epoch run whose first 1,600 steps are quoted shared it with unrelated work (load average up to about 40) and ran at the same speed per step.
- The memory and speed table comes from 3-step runs; numbers for a full run differ by a few percent.
