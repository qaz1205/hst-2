#!/usr/bin/env python3
"""
io_score.py — Surprisal scorer using the (under-trained) Io checkpoint.

The Io checkpoint at step ~1000 is far too early to generate coherent text,
but its forward pass is still a usable salience detector: tokens the model
finds *unlikely* given prior context tend to be the concrete / specific
content tokens of a sentence (proper nouns, numbers, technical terms),
while tokens it finds *likely* tend to be generic boilerplate ("the", "of",
"is", "this paper", etc.). By computing per-sentence cross-entropy in a
single batched forward pass we get a fast, content-agnostic salience signal
that we can blend into the existing BM25/Hebbian ranking.

Stdin schema:
{
  "checkpoint": str,
  "sentences": [str, ...],
  "max_tokens_per_sent": int = 48
}

Stdout (single JSON line):
{
  "scores": [float, ...],   # mean per-token NLL, higher = more surprising / specific
  "tokens": [int, ...],     # number of tokens scored per sentence
  "step": int
}
"""

import json
import os
import sys

import torch
import torch.nn.functional as F

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)

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
    if len(missing) > 50:
        _emit_error(f"Checkpoint mismatch: {len(missing)} missing keys")
    model.to(device)
    model.eval()
    step = int(ckpt.get("step", 0))
    return model, cfg, step


@torch.no_grad()
def score_batch(model, cfg, batch_ids):
    """
    batch_ids: list of token-id lists (already truncated). Returns list of
    (mean_nll, n_scored) tuples — one per sentence. NLL is computed only on
    the predicted tokens (positions 1..n-1), matching standard LM scoring.
    """
    device = next(model.parameters()).device
    pad_id = 0  # GPT-2 has no pad token; we mask via the lengths instead.
    max_len = max(len(s) for s in batch_ids)
    max_len = max(max_len, 2)

    B = len(batch_ids)
    inp = torch.full((B, max_len), pad_id, dtype=torch.long, device=device)
    lens = torch.zeros(B, dtype=torch.long, device=device)
    for i, ids in enumerate(batch_ids):
        n = len(ids)
        if n == 0:
            continue
        inp[i, :n] = torch.tensor(ids, dtype=torch.long, device=device)
        lens[i] = n

    out = model(inp, caches=None, training=False)
    logits = out["logits"]                # (B, T, V)
    log_probs = F.log_softmax(logits.float(), dim=-1)

    # Predict positions 1..n-1 from positions 0..n-2
    shift_lp = log_probs[:, :-1, :]                                   # (B, T-1, V)
    shift_tg = inp[:, 1:]                                             # (B, T-1)
    gathered = shift_lp.gather(2, shift_tg.unsqueeze(-1)).squeeze(-1) # (B, T-1)

    results = []
    for i in range(B):
        n = int(lens[i].item())
        scored = max(0, n - 1)
        if scored == 0:
            results.append((0.0, 0))
            continue
        nll = -gathered[i, :scored].mean().item()
        results.append((float(nll), scored))
    return results


def main():
    raw = sys.stdin.read()
    try:
        req = json.loads(raw)
    except Exception as e:
        _emit_error(f"Bad JSON: {e}")

    checkpoint = req.get("checkpoint")
    sentences = req.get("sentences", [])
    max_tok = int(req.get("max_tokens_per_sent", 48))

    if not checkpoint or not os.path.exists(checkpoint):
        _emit_error(f"Checkpoint not found: {checkpoint}")
    if not isinstance(sentences, list) or not sentences:
        print(json.dumps({"scores": [], "tokens": [], "step": 0}))
        return

    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
    except ImportError:
        _emit_error("tiktoken not installed")

    model, cfg, step = load_model(checkpoint)
    cap = min(max_tok, getattr(cfg, "max_seq_len", 512))

    encoded = []
    for s in sentences:
        s = (s or "").strip()
        if not s:
            encoded.append([enc.eot_token])  # placeholder so indexing aligns
            continue
        ids = enc.encode(s, allowed_special="all")[:cap]
        if len(ids) < 2:
            ids = ids + [enc.eot_token]
        encoded.append(ids)

    # Process in chunks to bound memory (B*T*V can be large on CPU)
    chunk = 8
    scores = []
    tokens = []
    for i in range(0, len(encoded), chunk):
        batch = encoded[i:i + chunk]
        for nll, n in score_batch(model, cfg, batch):
            scores.append(nll)
            tokens.append(n)

    print(json.dumps({"scores": scores, "tokens": tokens, "step": step}))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        _emit_error(f"Unhandled: {e}")
