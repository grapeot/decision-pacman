"""LoRA fine-tune the 0.8B student on Apple silicon with MLX.

  python training/train_mlx.py --data data/sft_q1 --out runs/ft_q1_mlx --epochs 2

This is the Mac counterpart of train.py and keeps its recipe: the same
training rows, the same chat-templated prompt (thinking disabled), the same
loss, schedule, batch order, and evaluation cadence. The loss reads the model
the way Ollama's /v1/systemone scorer does: the logits of the option letters
(A, B, C, ...) at the first answer position, softmax over those letters only,
cross-entropy against the row's target. Only that position is projected
through the output layer.

LoRA scaling: PEFT multiplies the update by alpha / rank, MLX by `scale`, so
rank 16 and alpha 32 become scale 2.0.

The adapter is written in mlx-lm's format (`adapters.safetensors` plus
`adapter_config.json`), so `mlx_lm.load(base, adapter_path=...)` reads it.
`merge_mlx.py` folds it into the Hugging Face checkpoint for GGUF export.
"""
import argparse
import json
import math
import os
import random
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map, tree_unflatten
from mlx_lm import load
from mlx_lm.tuner.lora import LoRALinear
from mlx_lm.tuner.trainer import grad_checkpoint

# Every language-side linear layer of Qwen3.5: attention, Gated DeltaNet, MLP (as in train.py).
TARGETS = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj",
    "gate_proj", "up_proj", "down_proj",
]


def unit_lower_inverse(L):
    """Inverse of I + L for strictly lower-triangular L [..., C, C], C a power of two.

    Recursive block inversion, bottom up: [[A, 0], [X, D]]^-1 = [[A^-1, 0], [-D^-1 X A^-1, D^-1]].
    Every intermediate is the inverse of a diagonal sub-block, so values stay as
    small as the answer's. (A Neumann-style product (I - L)(I + L^2)(I + L^4)...
    is exact in theory but overflows in float32 when keys repeat, as with padding.)
    """
    *pre, C, _ = L.shape
    M = L + mx.eye(C, dtype=L.dtype)
    inv = mx.ones((*pre, C, 1, 1), dtype=L.dtype)  # diagonal blocks of size 1
    bs = 1
    while bs < C:
        p = C // (2 * bs)  # pairs of diagonal blocks at this level
        # Below-diagonal block of pair i: rows (2i+1)bs..(2i+2)bs, columns 2i*bs..(2i+1)bs.
        rows = M.reshape(*pre, p, 2 * bs, C)[..., bs:, :]  # [..., p, bs, C]
        cols = rows.reshape(*pre, p, bs, p, 2 * bs)[..., :bs]  # [..., p, bs, p, bs]
        same = mx.eye(p, dtype=L.dtype)[:, None, :, None]  # keep column pair == row pair
        X = (cols * same).sum(axis=-2)  # [..., p, bs, bs]
        # Split the diagonal blocks into pairs through a reshape: in MLX 0.32.3 the
        # gradient of a step-2 slice (x[0::2]) over an axis of length 2 is wrong.
        pairs = inv.reshape(*pre, p, 2, bs, bs)
        A, D = pairs[..., 0, :, :], pairs[..., 1, :, :]
        top = mx.concatenate([A, mx.zeros_like(A)], axis=-1)
        bottom = mx.concatenate([-(D @ X @ A), D], axis=-1)
        inv = mx.concatenate([top, bottom], axis=-2)
        bs *= 2
    return inv.squeeze(-3)


def chunked_gated_delta(q, k, v, log_g, beta, state=None, chunk=64):
    """Gated delta rule over a whole sequence, chunk by chunk, in differentiable MLX ops.

    The same algorithm (UT transform within a chunk, a scan across chunks) as
    the chunked torch fallback in Hugging Face transformers, which is what the
    CUDA run used. mlx-lm's own training path loops over every token and keeps
    a [B, H, Dv, Dk] state per token for the backward pass, which is far slower
    and needs far more memory. The unit lower-triangular system of each chunk
    is inverted by unit_lower_inverse.

    Shapes follow mlx-lm: q, k [B, T, Hk, Dk] (q already scaled), v [B, T, Hv, Dv],
    log_g and beta [B, T, Hv], state [B, Hv, Dv, Dk]. Returns y [B, T, Hv, Dv], state.
    """
    B, T, Hk, Dk = q.shape
    Hv, Dv = v.shape[-2:]
    out_dtype = q.dtype
    if (rep := Hv // Hk) > 1:
        q, k = mx.repeat(q, rep, -2), mx.repeat(k, rep, -2)
    q, k, v = (x.astype(mx.float32).transpose(0, 2, 1, 3) for x in (q, k, v))  # [B, H, T, D]
    log_g, beta = (x.astype(mx.float32).transpose(0, 2, 1) for x in (log_g, beta))  # [B, H, T]
    pad = (chunk - T % chunk) % chunk
    if pad:
        q, k, v = (mx.pad(x, [(0, 0), (0, 0), (0, pad), (0, 0)]) for x in (q, k, v))
        log_g, beta = (mx.pad(x, [(0, 0), (0, 0), (0, pad)]) for x in (log_g, beta))
    n = (T + pad) // chunk
    k_beta, v_beta = k * beta[..., None], v * beta[..., None]
    q, k, k_beta, v_beta = (x.reshape(B, Hv, n, chunk, x.shape[-1]) for x in (q, k, k_beta, v_beta))
    cum = mx.cumsum(log_g.reshape(B, Hv, n, chunk), axis=-1)  # log of the decay since the chunk start
    lower = mx.tril(mx.ones((chunk, chunk), dtype=mx.bool_))
    strict = mx.tril(mx.ones((chunk, chunk), dtype=mx.bool_), -1)
    diff = cum[..., :, None] - cum[..., None, :]
    decay = mx.where(lower, mx.exp(mx.where(lower, diff, 0.0)), 0.0)  # exp(cum_i - cum_j) for j <= i
    kt = k.swapaxes(-1, -2)
    L = mx.where(strict, (k_beta @ kt) * decay, 0.0)
    inv = unit_lower_inverse(L)
    new_v = inv @ v_beta
    k_cum = inv @ (k_beta * mx.exp(cum)[..., None])
    intra = (q @ kt) * decay
    q_dec = q * mx.exp(cum)[..., None]
    k_dec = k * mx.exp(cum[..., -1:] - cum)[..., None]
    chunk_decay = mx.exp(cum[..., -1])[..., None, None]
    S = mx.zeros((B, Hv, Dk, Dv), dtype=mx.float32) if state is None else state.astype(mx.float32).swapaxes(-1, -2)
    outs = []
    for i in range(n):
        v_new = new_v[:, :, i] - k_cum[:, :, i] @ S
        outs.append(q_dec[:, :, i] @ S + intra[:, :, i] @ v_new)
        S = S * chunk_decay[:, :, i] + k_dec[:, :, i].swapaxes(-1, -2) @ v_new
    y = mx.stack(outs, axis=2).reshape(B, Hv, n * chunk, Dv)[:, :, :T]
    return y.transpose(0, 2, 1, 3).astype(out_dtype), S.swapaxes(-1, -2)


def use_chunked_training_path():
    """Route mlx-lm's Qwen3.5 training-mode gated delta call to chunked_gated_delta.

    Inference (use_kernel=True) still goes to mlx-lm's Metal kernel.
    """
    import mlx_lm.models.qwen3_5 as q35
    stock = q35.gated_delta_update

    def update(q, k, v, a, b, A_log, dt_bias, state=None, mask=None, *, use_kernel=True, lower_bound=None, allow_neg_eigval=False):
        if use_kernel or mask is not None or lower_bound is not None or allow_neg_eigval:
            return stock(q, k, v, a, b, A_log, dt_bias, state, mask, use_kernel=use_kernel,
                         lower_bound=lower_bound, allow_neg_eigval=allow_neg_eigval)
        log_g = -mx.exp(A_log.astype(mx.float32)) * nn.softplus(a + dt_bias)
        return chunked_gated_delta(q, k, v, log_g, mx.sigmoid(b), state)

    q35.gated_delta_update = update


def read_jsonl(path, limit=None):
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def apply_lora(model, rank, scale):
    """Wrap every target linear in every decoder layer; return the wrapped module paths."""
    keys = set()
    for layer in model.layers:
        swaps = []
        for name, mod in layer.named_modules():
            if name.split(".")[-1] in TARGETS and isinstance(mod, nn.Linear):
                swaps.append((name, LoRALinear.from_base(mod, r=rank, scale=scale, dropout=0.0)))
                keys.add(name)
        for name, new in swaps:
            parent = layer
            *path, leaf = name.split(".")
            for p in path:
                parent = getattr(parent, p)
            setattr(parent, leaf, new)
    return sorted(keys)


class Batcher:
    def __init__(self, tokenizer, letter_ids, max_len):
        self.tok, self.letter_ids, self.max_len = tokenizer, letter_ids, max_len
        self.pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    def prompt(self, row):
        return self.tok.apply_chat_template(row["messages"], tokenize=False, add_generation_prompt=True, enable_thinking=False)

    def tokenize(self, rows):
        for r in rows:
            r["ids"] = self.tok.encode(self.prompt(r), add_special_tokens=False)
            if len(r["ids"]) > self.max_len:
                raise ValueError(f"prompt of {len(r['ids'])} tokens exceeds --max-len {self.max_len}")

    def __call__(self, rows):
        width = max(len(r["ids"]) for r in rows)
        k = max(len(r["keys"]) for r in rows)
        ids, last, cand, valid, target = [], [], [], [], []
        for r in rows:
            n = len(r["keys"])
            ids.append(r["ids"] + [self.pad] * (width - len(r["ids"])))  # right padding, as in train.py
            last.append(len(r["ids"]) - 1)
            cand.append(self.letter_ids[:n] + [0] * (k - n))
            valid.append([True] * n + [False] * (k - n))
            target.append(list(r["target"]) + [0.0] * (k - n))
        return (mx.array(ids), mx.array(last), mx.array(cand), mx.array(valid),
                mx.array(target, dtype=mx.float32), sum(len(r["ids"]) for r in rows))


def out_weight(model):
    lm = model.language_model
    return lm.model.embed_tokens.weight if lm.args.tie_word_embeddings else lm.lm_head.weight


def option_logits(model, ids, last, cand):
    """Letter logits at each row's answer position: final hidden state times the letters' output rows."""
    hidden = model.language_model.model(ids)  # [B, T, H], after the final norm
    h = hidden[mx.arange(ids.shape[0]), last]  # [B, H]
    w = out_weight(model)[cand]  # [B, K, H]
    return (w.astype(mx.float32) @ h.astype(mx.float32)[:, :, None]).squeeze(-1)


def masked_log_softmax(logits, valid):
    return nn.log_softmax(mx.where(valid, logits, -1e9), axis=-1)


def loss_fn(model, ids, last, cand, valid, target):
    logp = masked_log_softmax(option_logits(model, ids, last, cand), valid)
    return -(target * mx.where(valid, logp, 0.0)).sum(axis=-1).mean()


def evaluate(model, batcher, rows, bs):
    model.eval()
    n = correct = 0
    ce_sum = 0.0
    for i in range(0, len(rows), bs):
        chunk = rows[i:i + bs]
        ids, last, cand, valid, target, _ = batcher(chunk)
        logp = masked_log_softmax(option_logits(model, ids, last, cand), valid)
        ce = -(target * mx.where(valid, logp, 0.0)).sum()
        hit = (logp.argmax(axis=-1) == target.argmax(axis=-1)).sum()
        mx.eval(ce, hit)
        ce_sum += ce.item()
        correct += hit.item()
        n += len(chunk)
    model.train()
    return {"ce": ce_sum / n, "acc": correct / n, "n": n}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--data", default="data/sft_q1")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=32, help="rows per optimizer step")
    ap.add_argument("--micro-batch", type=int, default=None, help="rows per forward/backward pass; gradients are accumulated up to --batch")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--val-limit", type=int, default=2000)
    ap.add_argument("--train-limit", type=int, default=None)
    ap.add_argument("--max-steps", type=int, default=None, help="stop early (for timing a subset); the schedule still spans the full run")
    ap.add_argument("--grad-checkpoint", action="store_true", help="recompute layer activations in the backward pass to save memory")
    ap.add_argument("--cache-limit-gb", type=float, default=4.0,
                    help="cap MLX's buffer cache; batches pad to different widths, so without a cap the cache grows by tens of GB per step")
    ap.add_argument("--resume", action="store_true", help="continue from <out>/checkpoint, written at every validation check")
    ap.add_argument("--recurrence", choices=["chunked", "stock"], default="chunked",
                    help="chunked: chunked_gated_delta for training (default); stock: mlx-lm's per-token loop")
    args = ap.parse_args()
    micro = args.micro_batch or args.batch
    if args.batch % micro:
        raise SystemExit("--batch must be a multiple of --micro-batch")
    accum = args.batch // micro
    os.makedirs(args.out, exist_ok=True)
    mx.random.seed(0)
    mx.set_cache_limit(int(args.cache_limit_gb * 1e9))

    if args.recurrence == "chunked":
        use_chunked_training_path()
    model, tok = load(args.base)
    model.freeze()
    letter_ids = []
    for ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        ids = tok.encode(ch, add_special_tokens=False)
        assert len(ids) == 1, f"letter {ch} is not a single token: {ids}"
        letter_ids.append(ids[0])

    scale = args.alpha / args.rank
    keys = apply_lora(model, args.rank, scale)
    print("LoRA targets:", keys, flush=True)
    n_train = sum(v.size for _, v in tree_flatten(model.trainable_parameters()))
    print(f"trainable parameters: {n_train:,}", flush=True)
    if args.grad_checkpoint:
        for cls in {type(layer) for layer in model.layers}:  # patches the class, so once per layer type
            grad_checkpoint(next(layer for layer in model.layers if type(layer) is cls))

    train = read_jsonl(os.path.join(args.data, "train.jsonl"), args.train_limit)
    val = read_jsonl(os.path.join(args.data, "val.jsonl"), args.val_limit)
    batcher = Batcher(tok, letter_ids, args.max_len)
    t_tok = time.time()
    batcher.tokenize(train)
    batcher.tokenize(val)
    lens = sorted(len(r["ids"]) for r in train)
    print(f"tokenized {len(train)} train / {len(val)} val rows in {time.time() - t_tok:.0f}s; "
          f"prompt tokens median {lens[len(lens) // 2]}, max {lens[-1]}", flush=True)
    print("example prompt:\n" + batcher.prompt(train[0]), flush=True)

    # Sanity check: the one-position projection must match the model's own last-position logits.
    ids, last, cand, valid, target, _ = batcher(train[:4])
    logits = option_logits(model, ids, last, cand)
    longest = int(mx.array([len(r["ids"]) for r in train[:4]]).argmax().item())
    full = model(ids[longest:longest + 1])[0, int(last[longest].item())].astype(mx.float32)
    diff = mx.abs(logits[longest] - full[cand[longest]]).max().item()
    print(f"projection check: max |diff| = {diff:.4f}", flush=True)
    assert diff < 0.5, "hidden-state projection does not match the model's logits"

    model.train()
    log_path = os.path.join(args.out, "log.jsonl")
    if not args.resume:
        t_eval = time.time()
        base_metrics = evaluate(model, batcher, val, micro)
        print(f"before training: {base_metrics} ({time.time() - t_eval:.0f}s)", flush=True)
        with open(log_path, "w") as f:
            f.write(json.dumps({"step": 0, **base_metrics}) + "\n")

    steps = math.ceil(len(train) * args.epochs / args.batch)
    warmup = max(1, int(0.03 * steps))

    def schedule(s):  # the LambdaLR of train.py: linear warmup times a cosine over the whole run
        s = s.astype(mx.float32)
        return args.lr * mx.minimum(1.0, (s + 1) / warmup) * 0.5 * (1 + mx.cos(math.pi * mx.minimum(1.0, s / steps)))

    def save_adapter():
        adapter = os.path.join(args.out, "adapter")
        os.makedirs(adapter, exist_ok=True)
        mx.save_safetensors(os.path.join(adapter, "adapters.safetensors"), dict(tree_flatten(model.trainable_parameters())))
        with open(os.path.join(adapter, "adapter_config.json"), "w") as f:
            json.dump({"model": args.base, "fine_tune_type": "lora", "num_layers": len(model.layers),
                       "lora_parameters": {"rank": args.rank, "scale": scale, "dropout": 0.0, "keys": keys}}, f, indent=2)
        return adapter

    # torch's AdamW always corrects the moment estimates for their zero start; MLX's does only when asked.
    # Without it the first few hundred updates are 3-6x larger than train.py's.
    opt = optim.AdamW(learning_rate=schedule, weight_decay=0.0, bias_correction=True)
    loss_and_grad = nn.value_and_grad(model, loss_fn)
    order = list(range(len(train)))
    rng = random.Random(0)
    pos = len(order)
    stop = min(steps, args.max_steps or steps)
    first, tokens, eval_seconds, elapsed, segments = 1, 0, 0.0, 0.0, 1
    ckpt = os.path.join(args.out, "checkpoint")

    def save_checkpoint(step):
        """Adapter, optimizer state, and data position, so --resume continues the same run."""
        save_adapter()
        os.makedirs(ckpt, exist_ok=True)
        mx.save_safetensors(os.path.join(ckpt, "optimizer.safetensors"), dict(tree_flatten(opt.state)))
        version, state, gauss = rng.getstate()
        with open(os.path.join(ckpt, "state.json"), "w") as f:
            json.dump({"step": step, "pos": pos, "order": order, "rng": [version, list(state), gauss], "tokens": tokens,
                       "eval_seconds": eval_seconds, "elapsed": time.time() - t0, "segments": segments}, f)

    if args.resume:
        with open(os.path.join(ckpt, "state.json")) as f:
            st = json.load(f)
        model.load_weights(os.path.join(args.out, "adapter", "adapters.safetensors"), strict=False)
        opt.init(model.trainable_parameters())
        opt.state = tree_unflatten(list(mx.load(os.path.join(ckpt, "optimizer.safetensors")).items()))
        order, pos, tokens, eval_seconds = st["order"], st["pos"], st["tokens"], st["eval_seconds"]
        rng.setstate((st["rng"][0], tuple(st["rng"][1]), st["rng"][2]))
        first, elapsed, segments = st["step"] + 1, st["elapsed"], st["segments"] + 1
        print(f"resumed after step {st['step']} ({elapsed:.0f}s so far)", flush=True)

    mx.reset_peak_memory()
    t0 = time.time() - elapsed  # wall time accumulates across resumed segments
    for step in range(first, stop + 1):
        if pos + args.batch > len(order):
            rng.shuffle(order)
            pos = 0
        rows = [train[i] for i in order[pos:pos + args.batch]]
        pos += args.batch
        grads = None
        loss = mx.array(0.0)
        for j in range(accum):
            ids, last, cand, valid, target, ntok = batcher(rows[j * micro:(j + 1) * micro])
            tokens += ntok
            l, g = loss_and_grad(model, ids, last, cand, valid, target)
            if accum > 1:
                grads = g if grads is None else tree_map(lambda x, y: x + y, grads, g)
                mx.eval(grads, l)
            else:
                grads = g
            loss = loss + l / accum
        if accum > 1:
            grads = tree_map(lambda x: x / accum, grads)
        grads, _ = optim.clip_grad_norm(grads, 1.0)
        opt.update(model, grads)
        mx.eval(model.trainable_parameters(), opt.state, loss)
        if step % args.log_every == 0 or step < first + 3:
            el = time.time() - t0
            train_s = el - eval_seconds
            print(f"step {step}/{steps} loss {loss.item():.4f} lr {schedule(mx.array(step)).item():.2e} {el:.0f}s "
                  f"{tokens / train_s:.0f} tok/s active {mx.get_active_memory() / 1e9:.1f} GB cache {mx.get_cache_memory() / 1e9:.1f} GB peak {mx.get_peak_memory() / 1e9:.1f} GB", flush=True)
        if step % args.eval_every == 0 or step == steps:
            te = time.time()
            m = evaluate(model, batcher, val, micro)
            eval_seconds += time.time() - te
            print(f"eval step {step}: {m} ({time.time() - te:.0f}s)", flush=True)
            with open(log_path, "a") as f:
                f.write(json.dumps({"step": step, "loss": loss.item(), **m}) + "\n")
            save_checkpoint(step)  # a crash or kill loses at most --eval-every steps

    wall = time.time() - t0
    summary = {"steps": stop, "steps_planned": steps, "wall_seconds": round(wall, 1), "eval_seconds": round(eval_seconds, 1),
               "train_seconds": round(wall - eval_seconds, 1), "prompt_tokens": tokens,
               "tokens_per_second": round(tokens / (wall - eval_seconds), 1),
               "peak_memory_gb": round(mx.get_peak_memory() / 1e9, 2), "segments": segments, "batch": args.batch, "micro_batch": micro,
               "grad_checkpoint": args.grad_checkpoint, "recurrence": args.recurrence, "cache_limit_gb": args.cache_limit_gb, "rank": args.rank, "scale": scale, "lr": args.lr,
               "optimizer": "AdamW, bias correction, weight decay 0"}
    print("summary:", json.dumps(summary), flush=True)
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("saved adapter ->", save_adapter(), flush=True)


if __name__ == "__main__":
    main()
