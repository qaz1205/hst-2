#!/usr/bin/env python3
"""
io_infer_v2.py — Io neural model inference bridge (advanced).

Reads a single JSON object from stdin describing the request, loads the
checkpoint ONCE, draws N samples, and prints one JSON object on stdout.

Supports every Io sampling knob: temperature, top_k, top_p, repetition_penalty,
presence_penalty, n_samples (best-of-N in a single model load), max_tokens,
stop_token (defaults to GPT-2 EOS = 50256), seed.

Stdin schema:
{
  "prompt": str,
  "checkpoint": str,
  "max_tokens": int = 200,
  "temperature": float = 0.85,
  "top_k": int = 50,
  "top_p": float = 0.95,
  "repetition_penalty": float = 1.15,
  "presence_penalty": float = 0.0,
  "n_samples": int = 1,
  "seed": int | null = null
}

Stdout (last line, single JSON object):
{
  "samples": [{"text": str, "logprob": float, "tokens": int}, ...],
  "step": int,
  "checkpoint": str
}
"""

import json
import os
import sys
import math
import random

import torch
import torch.nn.functional as F

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)

# io_train.py exports `Io` and `IoConfig`. The original io_infer expected aliases
# `IoCrystallineModel` / `TrainConfig` that don't exist — we use the real names.
try:
    from io_train_1778915806650 import Io as _IoModel, IoConfig as _IoConfig  # type: ignore
except ImportError:
    try:
        from io_train import Io as _IoModel, IoConfig as _IoConfig  # type: ignore
    except ImportError as e:
        print(json.dumps({"error": f"Could not import io_train: {e}"}))
        sys.exit(1)


def _emit_error(msg: str) -> None:
    print(json.dumps({"error": msg}))
    sys.exit(1)


def load_model(checkpoint_path: str, device: str = "cpu"):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg_dict = ckpt.get("config", {})
    cfg = _IoConfig()
    for k, v in cfg_dict.items():
        if hasattr(cfg, k):
            try:
                setattr(cfg, k, v)
            except Exception:
                pass
    cfg.device = device
    cfg.use_amp = False

    model = _IoModel(cfg)
    state = ckpt.get("model_state", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    # Tolerate small key drift from architecture revisions; fail only on a
    # catastrophic mismatch.
    if len(missing) > 50:
        _emit_error(f"Checkpoint mismatch: {len(missing)} missing keys")
    model.to(device)
    model.eval()
    step = int(ckpt.get("step", 0))
    return model, cfg, step


@torch.no_grad()
def _sample_one(
    model,
    cfg,
    prompt_tokens,
    max_new: int,
    temperature: float,
    top_k: int,
    top_p: float,
    rep_penalty: float,
    presence_penalty: float,
    eos: int = 50256,
):
    device = next(model.parameters()).device
    seq = list(prompt_tokens[-cfg.max_seq_len:])
    generated = []
    seen = {}  # token id -> count, for repetition / presence penalties
    for t in prompt_tokens[-256:]:  # seed presence with recent prompt context
        seen[t] = seen.get(t, 0) + 1

    sum_logprob = 0.0

    for _ in range(max_new):
        x = torch.tensor([seq], dtype=torch.long, device=device)
        out = model(x, caches=None, training=False)
        logits = out["logits"][0, -1, :].clone()  # (V,)

        # Repetition penalty: divide logits of seen tokens (positive logits)
        # or multiply (negative) per the standard HF implementation.
        if rep_penalty and rep_penalty != 1.0 and seen:
            ids = torch.tensor(list(seen.keys()), dtype=torch.long, device=device)
            sel = logits.index_select(0, ids)
            sel = torch.where(sel > 0, sel / rep_penalty, sel * rep_penalty)
            logits.index_copy_(0, ids, sel)

        # Presence penalty: flat subtract from already-seen tokens
        if presence_penalty and seen:
            ids = torch.tensor(list(seen.keys()), dtype=torch.long, device=device)
            logits.index_add_(0, ids, torch.full_like(ids, -float(presence_penalty), dtype=logits.dtype))

        # Temperature
        logits = logits / max(temperature, 1e-6)

        # Top-k
        if top_k and top_k > 0:
            v, _idx = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[-1]] = float("-inf")

        # Top-p (nucleus)
        if top_p and 0.0 < top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            sorted_probs = F.softmax(sorted_logits, dim=-1)
            cumprobs = torch.cumsum(sorted_probs, dim=-1)
            keep = cumprobs - sorted_probs <= top_p
            keep[0] = True  # always keep top-1
            sorted_logits[~keep] = float("-inf")
            logits = torch.full_like(logits, float("-inf"))
            logits.scatter_(0, sorted_idx, sorted_logits)

        probs = F.softmax(logits, dim=-1)
        if not torch.isfinite(probs).all() or probs.sum().item() <= 0:
            # Degenerate distribution — fall back to argmax of original logits
            next_tok = int(out["logits"][0, -1, :].argmax().item())
        else:
            next_tok = int(torch.multinomial(probs, num_samples=1).item())

        # Track logprob for ranking
        lp = float(torch.log(probs[next_tok].clamp(min=1e-12)).item())
        sum_logprob += lp

        if next_tok == eos:
            break

        generated.append(next_tok)
        seen[next_tok] = seen.get(next_tok, 0) + 1
        seq.append(next_tok)
        if len(seq) >= cfg.max_seq_len:
            seq = seq[1:]

    return generated, sum_logprob


def main():
    raw = sys.stdin.read()
    if not raw.strip():
        _emit_error("No JSON request on stdin")
    try:
        req = json.loads(raw)
    except json.JSONDecodeError as e:
        _emit_error(f"Bad JSON: {e}")

    prompt = req.get("prompt") or ""
    checkpoint = req.get("checkpoint") or ""
    if not prompt or not checkpoint:
        _emit_error("prompt and checkpoint are required")
    if not os.path.isfile(checkpoint):
        _emit_error(f"Checkpoint not found: {checkpoint}")

    max_tokens = int(req.get("max_tokens", 200))
    max_tokens = max(1, min(max_tokens, 1024))
    temperature = float(req.get("temperature", 0.85))
    top_k = int(req.get("top_k", 50))
    top_p = float(req.get("top_p", 0.95))
    rep_penalty = float(req.get("repetition_penalty", 1.15))
    presence_penalty = float(req.get("presence_penalty", 0.0))
    n_samples = max(1, min(int(req.get("n_samples", 1)), 8))
    seed = req.get("seed")
    if seed is not None:
        try:
            torch.manual_seed(int(seed))
            random.seed(int(seed))
        except Exception:
            pass

    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
    except ImportError:
        _emit_error("tiktoken not installed")

    try:
        model, cfg, step = load_model(checkpoint)
    except Exception as e:
        _emit_error(f"Failed to load checkpoint: {e}")

    prompt_tokens = enc.encode(prompt, allowed_special="all")

    samples = []
    for i in range(n_samples):
        # Vary seed slightly per sample so multinomial produces different draws
        if seed is not None:
            torch.manual_seed(int(seed) + i)
        try:
            tokens, lp = _sample_one(
                model, cfg, prompt_tokens,
                max_tokens, temperature, top_k, top_p,
                rep_penalty, presence_penalty,
            )
            text = enc.decode(tokens)
        except Exception as e:
            text, lp, tokens = "", -1e9, []
            print(f"[infer_v2] sample {i} failed: {e}", file=sys.stderr)
        samples.append({
            "text": text,
            "logprob": lp,
            "tokens": len(tokens),
            "avg_logprob": (lp / max(len(tokens), 1)),
        })

    print(json.dumps({
        "samples": samples,
        "step": step,
        "checkpoint": os.path.basename(checkpoint),
    }))


if __name__ == "__main__":
    main()
