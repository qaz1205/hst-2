#!/usr/bin/env python3
"""
io_finetune.py — Light CPU fine-tuning of an Io checkpoint on a local corpus.

Reads a JSON request from stdin:
{
  "checkpoint_in":  "io_checkpoints/io_step_001000.pt",
  "checkpoint_out": "io_checkpoints/io_step_001000_ft.pt",
  "corpus_path":    "artifacts/api-server/data/finetune_corpus.txt",
  "steps":          50,
  "lr":             1.0e-5,
  "seq_len":        256,
  "batch_size":     1,
  "grad_clip":      1.0
}

Emits one JSON line per training step on stdout (so the API can stream progress)
plus a final {"done": true, ...} line.
"""

import json
import os
import sys
import time
import math
import random
from typing import List

import torch
import torch.nn.functional as F

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)

try:
    from io_train_1778915806650 import Io as _IoModel, IoConfig as _IoConfig  # type: ignore
except ImportError:
    from io_train import Io as _IoModel, IoConfig as _IoConfig  # type: ignore


def _emit(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def load_corpus(path: str, encode_fn, seq_len: int, max_chunks: int = 256) -> List[List[int]]:
    """Read corpus, tokenize, slice into seq_len+1 chunks (last token = next-token target)."""
    if not os.path.isfile(path):
        return []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    if not text.strip():
        return []
    tokens = encode_fn(text)
    sl1 = seq_len + 1
    chunks: List[List[int]] = []
    for i in range(0, len(tokens) - sl1, sl1):
        chunks.append(tokens[i:i + sl1])
        if len(chunks) >= max_chunks:
            break
    return chunks


def main():
    raw = sys.stdin.read()
    if not raw.strip():
        _emit({"error": "No JSON request on stdin"})
        sys.exit(1)
    req = json.loads(raw)

    ckpt_in  = req["checkpoint_in"]
    ckpt_out = req["checkpoint_out"]
    corpus   = req["corpus_path"]
    steps    = max(1, min(int(req.get("steps", 30)), 500))
    lr       = float(req.get("lr", 1e-5))
    seq_len  = int(req.get("seq_len", 256))
    batch    = max(1, int(req.get("batch_size", 1)))
    clip     = float(req.get("grad_clip", 1.0))

    if not os.path.isfile(ckpt_in):
        _emit({"error": f"checkpoint_in missing: {ckpt_in}"})
        sys.exit(1)

    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
    except ImportError:
        _emit({"error": "tiktoken not installed"})
        sys.exit(1)

    _emit({"phase": "load_checkpoint", "path": ckpt_in})
    ckpt = torch.load(ckpt_in, map_location="cpu", weights_only=False)
    cfg_dict = ckpt.get("config", {})
    cfg = _IoConfig()
    for k, v in cfg_dict.items():
        if hasattr(cfg, k):
            try: setattr(cfg, k, v)
            except: pass
    cfg.device = "cpu"
    cfg.use_amp = False
    cfg.batch_size = batch
    # Do NOT override max_seq_len: it controls RoPE cache shape inside the
    # model, and overriding it makes the checkpoint state_dict incompatible.
    # We just slice training samples to `seq_len` instead.
    if seq_len > cfg.max_seq_len:
        seq_len = cfg.max_seq_len

    model = _IoModel(cfg)
    state = ckpt.get("model_state", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if len(missing) > 50:
        _emit({"error": f"too many missing keys: {len(missing)}"})
        sys.exit(1)
    model.to("cpu")
    model.train()

    n_params = sum(p.numel() for p in model.parameters())
    _emit({"phase": "model_loaded", "params": n_params, "missing_keys": len(missing)})

    _emit({"phase": "load_corpus", "path": corpus})
    encode = lambda t: enc.encode(t, allowed_special="all")
    chunks = load_corpus(corpus, encode, seq_len)
    if not chunks:
        _emit({"error": f"empty corpus: {corpus}"})
        sys.exit(1)
    _emit({"phase": "corpus_loaded", "chunks": len(chunks), "tokens_total": sum(len(c) for c in chunks)})

    # Train only the final layer norm + LM head + last block, freeze the rest.
    # Minimises CPU compute; still adapts the output distribution.
    for n, p in model.named_parameters():
        p.requires_grad = False
    train_keywords = ("blocks.11", "ln_f", "head", "lm_head", "out_proj", "embed")
    trainables = [p for n, p in model.named_parameters()
                  if any(k in n for k in train_keywords)]
    for p in trainables:
        p.requires_grad = True
    n_train = sum(p.numel() for p in trainables)
    _emit({"phase": "params_unfrozen", "trainable": n_train, "total": n_params,
           "fraction": round(n_train / max(n_params, 1), 4)})

    optimizer = torch.optim.AdamW(trainables, lr=lr, weight_decay=0.01, betas=(0.9, 0.95))

    losses = []
    t0 = time.time()
    for step in range(1, steps + 1):
        # Sample a random chunk (or stack `batch` of them)
        sample = random.sample(chunks, min(batch, len(chunks)))
        x = torch.tensor([c[:seq_len]   for c in sample], dtype=torch.long)
        y = torch.tensor([c[1:seq_len+1] for c in sample], dtype=torch.long)

        optimizer.zero_grad(set_to_none=True)
        try:
            out = model(x, caches=None, training=True)
            logits = out["logits"]
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                y.reshape(-1),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainables, clip)
            optimizer.step()
            l = float(loss.item())
        except Exception as e:
            _emit({"phase": "step_error", "step": step, "error": str(e)[:200]})
            continue

        losses.append(l)
        elapsed = time.time() - t0
        _emit({
            "phase": "step",
            "step": step,
            "total_steps": steps,
            "loss": round(l, 4),
            "elapsed_s": round(elapsed, 1),
            "tok_per_s": round((step * seq_len * batch) / max(elapsed, 1e-6), 1),
        })

    avg = sum(losses) / max(len(losses), 1)
    _emit({"phase": "save", "path": ckpt_out, "avg_loss": round(avg, 4)})
    os.makedirs(os.path.dirname(ckpt_out) or ".", exist_ok=True)
    # Atomic write: torch.save to .tmp, fsync, then rename. Prevents inference
    # from picking up a half-written checkpoint while we're still saving.
    tmp_out = ckpt_out + ".tmp"
    torch.save({
        "step": int(ckpt.get("step", 1000)) + steps,
        "model_state": model.state_dict(),
        "config": cfg.to_dict() if hasattr(cfg, "to_dict") else cfg_dict,
        "version": "3.2-ft",
        "base_step": int(ckpt.get("step", 1000)),
        "ft_steps": steps,
        "ft_loss_avg": avg,
        "ft_corpus": os.path.basename(corpus),
    }, tmp_out)
    try:
        with open(tmp_out, "rb") as fh:
            os.fsync(fh.fileno())
    except OSError:
        pass
    os.replace(tmp_out, ckpt_out)

    size_mb = os.path.getsize(ckpt_out) / 1024 / 1024
    _emit({"done": True, "checkpoint_out": ckpt_out, "size_mb": round(size_mb, 1),
           "avg_loss": round(avg, 4), "steps": steps,
           "elapsed_s": round(time.time() - t0, 1)})


if __name__ == "__main__":
    main()
