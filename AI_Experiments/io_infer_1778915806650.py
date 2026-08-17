#!/usr/bin/env python3
"""
io_infer.py — Io neural model inference bridge.

Loads a checkpoint produced by io_train.py, runs greedy generation
given a prompt, and prints a single JSON object to stdout.

Usage:
  python3 io/io_infer.py \
      --prompt "What is quantum entanglement?" \
      --checkpoint io/io_checkpoints/io_step_001000.pt \
      --max_tokens 120

Output (stdout, one line):
  {"text": "...", "step": 1000, "checkpoint": "io_step_001000.pt"}
"""

import argparse
import json
import os
import sys
import types

import torch
import torch.nn.functional as F

# ── locate io_train.py and import the model class ──────────────────────────
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)

# Import only what we need from the training script
try:
    from io_train import IoCrystallineModel, TrainConfig
except ImportError as e:
    print(json.dumps({"error": f"Could not import io_train: {e}"}))
    sys.exit(1)


def load_model(checkpoint_path: str, device: str = "cpu"):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg_dict = ckpt.get("config", {})

    # Rebuild config
    cfg = TrainConfig()
    for k, v in cfg_dict.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)

    model = IoCrystallineModel(cfg)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.to(device)
    model.eval()

    step = ckpt.get("step", 0)
    return model, cfg, step


@torch.no_grad()
def generate(model, cfg, prompt_tokens: list[int], max_new: int, temperature: float = 0.8, top_k: int = 50) -> list[int]:
    device = next(model.parameters()).device
    seq = list(prompt_tokens[-cfg.max_seq_len:])

    generated = []
    for _ in range(max_new):
        x = torch.tensor([seq], dtype=torch.long, device=device)
        out = model(x, caches=None, training=False)

        logits = out["logits"][0, -1, :]  # (vocab_size,)

        # Top-k + temperature sampling
        logits = logits / max(temperature, 1e-6)
        if top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[-1]] = float("-inf")

        probs = F.softmax(logits, dim=-1)
        next_tok = torch.multinomial(probs, num_samples=1).item()

        if next_tok == 50256:  # EOS
            break

        generated.append(next_tok)
        seq.append(next_tok)
        if len(seq) >= cfg.max_seq_len:
            seq = seq[1:]

    return generated


def main():
    parser = argparse.ArgumentParser(description="Io inference bridge")
    parser.add_argument("--prompt",      required=True, help="Text prompt")
    parser.add_argument("--checkpoint",  required=True, help="Path to .pt checkpoint")
    parser.add_argument("--max_tokens",  type=int, default=120, help="Max new tokens")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k",       type=int, default=50)
    args = parser.parse_args()

    if not os.path.isfile(args.checkpoint):
        print(json.dumps({"error": f"Checkpoint not found: {args.checkpoint}"}))
        sys.exit(1)

    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
    except ImportError:
        print(json.dumps({"error": "tiktoken not installed"}))
        sys.exit(1)

    try:
        model, cfg, step = load_model(args.checkpoint)
    except Exception as e:
        print(json.dumps({"error": f"Failed to load checkpoint: {e}"}))
        sys.exit(1)

    prompt_tokens = enc.encode(args.prompt, allowed_special="all")
    try:
        token_ids = generate(model, cfg, prompt_tokens, args.max_tokens, args.temperature, args.top_k)
        text = enc.decode(token_ids)
    except Exception as e:
        print(json.dumps({"error": f"Generation failed: {e}"}))
        sys.exit(1)

    ckpt_name = os.path.basename(args.checkpoint)
    print(json.dumps({"text": text, "step": step, "checkpoint": ckpt_name}))


if __name__ == "__main__":
    main()
