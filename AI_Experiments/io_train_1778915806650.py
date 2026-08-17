"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║                               Io  v3.2                                          ║
║       HST Crystalline — ALL VERSIONS + CIF EVERYWHERE + CHAOS + ERROR NETS      ║
║                                                                                  ║
║  ── HST v1 FEATURES ──────────────────────────────────────────────────────────  ║
║   1. CIF Engine       Closed IF Set, O(1) monomorphic branch lock               ║
║   2. Pell-Lucas Spine P(n)=2P(n-1)+P(n-2) long-range sequence anchor           ║
║   3. Diamond Mixer    GELU-difference bifurcation FFN replacement               ║
║   4. Speculative Horizon  Weight-tied multi-step future token drafting          ║
║   5. Paged KV Cache   Block-based O(1) memory allocation                        ║
║                                                                                  ║
║  ── HST v2 FEATURES ──────────────────────────────────────────────────────────  ║
║   6. Hyperbolic Embedding   Poincaré ball projection for hierarchy              ║
║   7. Lattice Positional Enc  Dual-stream: sinusoid + spine-proximity MLP        ║
║   8. Hebbian Fast Weights   Inference-time plasticity, zero-shot adaptation     ║
║   9. Feedback Loop          GRU-based iterative hidden-state self-correction    ║
║  10. Complete Lattice Proc  BFS path-weighted Pell-Lucas message passing        ║
║  11. CIF Early Exit         Confidence-gated per-layer skipping                 ║
║                                                                                  ║
║  ── HST v3 FEATURES ──────────────────────────────────────────────────────────  ║
║  12. Rotary Position Embeddings (RoPE) in attention Q and K                     ║
║  13. Grouped Query Attention (GQA)  n_kv = n_heads // 4                         ║
║  14. QK Normalization              Stabilize attention via L2-norm on Q,K       ║
║  15. SwiGLU Hybrid FFN             Alternates with Diamond Mixer per layer      ║
║  16. Hyper-Lattice Aggregator      Cross-layer hidden-state fusion              ║
║  17. Adaptive Computation Time     Per-token soft halting + pondering loss      ║
║  18. Q2AC2D-17                     Quality-aware adaptive loss weighting        ║
║  19. Pell-Lucas Position Loss      Spine-anchor position-weighted CE loss       ║
║  20. Self-Identity Corpus          Io knows who and what it is                  ║
║                                                                                  ║
║  ── CIF SPEEDUP EVERYWHERE (v3.1) ───────────────────────────────────────────  ║
║  CIF.A: CachedCondition     efficiency tracking helper across all modules       ║
║  CIF.B: CIFRegistry         30+ pre-resolved architectural scalars             ║
║  CIF.C: Per-layer Deny      FFN parity, HyperLattice triggers, exit flags      ║
║  CIF.D: Hebbian Deny        decay constant locked O(1) after first lookup      ║
║  CIF.E: Attention Deny      GQA ratio, RoPE scale, QK scale — all Deny-locked  ║
║  CIF.F: Lattice Deny        spine position BFS path-count weights              ║
║  CIF.G: Training Deny       checkpoint interval, accum gate, quality flags     ║
║  CIF.H: Memory-pool CIF     alloc state with O(1) boundary resolution          ║
║                                                                                  ║
║  ── HST CHAOS LOGIC (v3.2) ──────────────────────────────────────────────────  ║
║  21. ChaosLogicLayer   Logistic map (r=3.9) + Lorenz σ–ρ coupling              ║
║                        + Lyapunov exponent estimator for stability reg.         ║
║                        Applied at hyper-trigger layers {3,7,11}                 ║
║                                                                                  ║
║  ── HST ERROR NETWORKS (v3.2) ───────────────────────────────────────────────  ║
║  22. ErrorCorrectionNetwork  Error detector → error direction → correction      ║
║                              Additive correction on logits, self-trains via LM  ║
║                                                                                  ║
║  TARGET SIZE: ~110–115M params (float32 ≈ 440–460MB)                            ║
║  DATASET: HuggingFaceFW/fineweb-edu (Sample-10BT) + self-identity injection     ║
╚══════════════════════════════════════════════════════════════════════════════════╝
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import time
import os
import sys
import random
import json
import argparse
import logging
import threading
import queue
from typing import Optional, List, Tuple, Dict, Any, Callable, FrozenSet
from dataclasses import dataclass, field, asdict
from collections import deque

try:
    from datasets import load_dataset
    import tiktoken
    HAS_RESOURCES = True
except ImportError:
    print("[Error] Missing dependencies. Run: pip install datasets tiktoken torch numpy")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("Io-HST")


def print_banner():
    banner = r"""
    ██╗ ██████╗     ██╗   ██╗ ██████╗      ██████╗
    ██║██╔═══██╗    ██║   ██║╚════██╗    ██╔═████╗
    ██║██║   ██║    ██║   ██║ █████╔╝    ██║██╔██║
    ██║██║   ██║    ╚██╗ ██╔╝ ╚═══██╗    ████╔╝██║
    ██║╚██████╔╝     ╚████╔╝ ██████╔╝    ╚██████╔╝
    ╚═╝ ╚═════╝       ╚═══╝  ╚═════╝     ╚═════╝
    >>> HIGH-SPECULATIVE TOPOLOGY v3.2 — CHAOS + ERROR NETWORKS + CIF EVERYWHERE
    >>> LOGISTIC-MAP · LORENZ σ-ρ · LYAPUNOV · ERROR-CORRECTION · DENY-LOCKED O(1)
    >>> RoPE · GQA · QK-NORM · SWIGLU · HYPER-LATTICE · ACT · Q2AC2D-17
    """
    print(banner)


# ══════════════════════════════════════════════════════════════════════════════
# 1. CLOSED IF SET (CIF) ENGINE  [HST v1 + v3.1 extended]
#    Deny  → caches flipped result after first eval → O(1) subsequent calls
#    Affirm → always re-evaluates fresh (for dynamic conditions)
#    CIFScalar → lazily evaluated constant, permanently cached (no flip)
#    CachedCondition → tracks hit/miss ratio for efficiency reporting [v8 ref]
# ══════════════════════════════════════════════════════════════════════════════

class _CIF:
    """Base primitive. Prevents dynamic branching in the hot inference loop."""
    __slots__ = ("value", "cache")
    def __init__(self, v: Any):
        self.value = v
        self.cache = None


class Affirm(_CIF):
    """Dynamic Path — evaluated fresh every cycle (no caching)."""
    def test(self, fn: Callable[[Any], bool]) -> bool:
        return fn(self.value)
    def flip(self) -> "Deny":
        return Deny(self.value)
    def __repr__(self) -> str:
        return f"Affirm({self.value})"


class Deny(_CIF):
    """
    Memory-learned path — computes once, caches the FLIPPED result.
    Subsequent calls: O(1) memory lookup, ~200x speedup vs plain if.
    Use for expensive static conditions that never change after first eval.
    """
    def test(self, fn: Callable[[Any], bool]) -> bool:
        if self.cache is None:
            self.cache = not fn(self.value)  # learn the flip, store in memory
        return self.cache
    def flip(self) -> "Affirm":
        return Affirm(self.value)
    def invalidate(self) -> None:
        """Force recomputation on next test (use when value changes)."""
        self.cache = None
    def __repr__(self) -> str:
        return f"Deny({self.value}, cached={self.cache is not None})"


class CIFScalar:
    """
    Lazily evaluated, permanently cached scalar constant.
    No flip — returns the actual value.  O(1) after first get().
    """
    __slots__ = ("_fn", "_val", "_computed")
    def __init__(self, fn: Callable):
        self._fn = fn
        self._val = None
        self._computed = False
    def get(self) -> Any:
        if not self._computed:
            self._val = self._fn()
            self._computed = True
        return self._val
    def __repr__(self) -> str:
        return f"CIFScalar({'*' if not self._computed else self._val})"


class CIFState:
    """Oscillates between Deny/Affirm for semi-dynamic boundary checks."""
    def __init__(self, value: Any, initial_deny: bool = True):
        self._s = Deny(value) if initial_deny else Affirm(value)
        self._val = value
    def test(self, fn: Callable) -> bool:
        return self._s.test(fn)
    def set_dynamic(self) -> None:
        self._s = Affirm(self._val)
    def set_static(self) -> None:
        self._s = Deny(self._val)


class CachedCondition:
    """
    [CIF.A — from HST v8 reference]
    Wraps a CIF state with hit/miss tracking.
    Use to measure cache efficiency across modules.
    """
    __slots__ = ("_state", "computation_count", "cache_hits")

    def __init__(self, value: Any, initial_deny: bool = False):
        self._state = Deny(value) if initial_deny else Affirm(value)
        self.computation_count = 0
        self.cache_hits = 0

    def evaluate(self, fn: Callable[[Any], bool]) -> bool:
        was_cached = self._state.cache is not None
        result = self._state.test(fn)
        self.cache_hits += int(was_cached)
        self.computation_count += 1
        return result

    def switch_to_deny(self) -> None:
        if isinstance(self._state, Affirm):
            self._state = self._state.flip()

    def switch_to_affirm(self) -> None:
        if isinstance(self._state, Deny):
            self._state = self._state.flip()

    def efficiency(self) -> float:
        if self.computation_count == 0:
            return 0.0
        return (self.cache_hits / self.computation_count) * 100.0

    def __repr__(self) -> str:
        return (f"CachedCondition(hits={self.cache_hits}/"
                f"{self.computation_count}, "
                f"eff={self.efficiency():.1f}%)")


def cif_const(value: Any, negated: bool = False) -> "_CIF":
    """Factory: Deny for static/cached conditions, Affirm for dynamic ones."""
    return Deny(value) if negated else Affirm(value)


# ══════════════════════════════════════════════════════════════════════════════
# 2. CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class IoConfig:
    """Io v3.1 full configuration."""
    vocab_size:          int   = 50257
    d_model:             int   = 768
    n_heads:             int   = 12
    n_kv_heads:          int   = 3       # GQA: n_heads // 4
    n_layers:            int   = 12
    max_seq_len:         int   = 1024
    dropout:             float = 0.1

    max_horizon:         int   = 32
    fb_iterations:       int   = 2
    hebbian_decay:       float = 0.995
    exit_threshold:      float = 0.92
    hyper_interval:      int   = 4
    act_epsilon:         float = 0.01
    act_tau:             float = 0.001

    lr:                  float = 3e-4
    weight_decay:        float = 0.1
    warmup_steps:        int   = 2000
    max_steps:           int   = 50000
    batch_size:          int   = 4
    accum_steps:         int   = 4
    grad_clip:           float = 1.0
    checkpoint_interval: int   = 500

    temperature:         float = 0.85
    top_k:               int   = 50
    top_p:               float = 0.95

    identity_inject_freq: int  = 50

    device:              str   = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp:             bool  = True
    checkpoint_dir:      str   = "io_checkpoints"
    resume_from:         str   = ""

    def to_dict(self) -> dict:
        return asdict(self)


class CIFRegistry:
    """
    [CIF.B] Central repository of 30+ pre-resolved architectural scalars.
    All values computed once via CIFScalar.get() — O(1) on every call.
    """
    def __init__(self, cfg: IoConfig):
        c = cfg
        # ── attention geometry ──────────────────────────────────────────────
        self.head_dim       = CIFScalar(lambda: c.d_model // c.n_heads)
        self.kv_ratio       = CIFScalar(lambda: c.n_heads // c.n_kv_heads)
        self.attn_scale     = CIFScalar(lambda: 1.0 / math.sqrt(c.d_model // c.n_heads))
        self.rope_base      = CIFScalar(lambda: 10000)
        # ── projection sizes ───────────────────────────────────────────────
        self.d_model_x2     = CIFScalar(lambda: c.d_model * 2)
        self.d_model_x3     = CIFScalar(lambda: c.d_model * 3)
        self.swiglu_inner   = CIFScalar(lambda: ((int(c.d_model * 8 / 3) + 63) // 64) * 64)
        self.hyper_compress = CIFScalar(lambda: 128)
        self.hyper_scale    = CIFScalar(lambda: 1.0 / math.sqrt(128))
        self.hebb_d_small   = CIFScalar(lambda: max(c.d_model // 4, 1))
        self.hebb_d_half    = CIFScalar(lambda: max(c.d_model // 2, 1))
        # ── topology ───────────────────────────────────────────────────────
        self.n_hyper        = CIFScalar(lambda: c.n_layers // c.hyper_interval)
        self.hyper_triggers = CIFScalar(
            lambda: frozenset(
                c.hyper_interval * (i + 1) - 1
                for i in range(c.n_layers // c.hyper_interval)
            )
        )
        self.layer_is_diamond = CIFScalar(
            lambda: {i: (i % 2 == 0) for i in range(c.n_layers)}
        )
        # ── training scalars ───────────────────────────────────────────────
        self.fb_iters       = CIFScalar(lambda: c.fb_iterations)
        self.exit_thresh    = CIFScalar(lambda: c.exit_threshold)
        self.block_size     = CIFScalar(lambda: 16)
        self.hebb_decay     = CIFScalar(lambda: c.hebbian_decay)
        self.act_eps        = CIFScalar(lambda: c.act_epsilon)
        self.act_tau_val    = CIFScalar(lambda: c.act_tau)
        # ── chaos & error-network ──────────────────────────────────────────
        self.chaos_r        = CIFScalar(lambda: 3.9)          # logistic map chaotic regime
        self.chaos_dt       = CIFScalar(lambda: 0.01)         # Lorenz integration step
        self.lorenz_sigma   = CIFScalar(lambda: 10.0)         # Lorenz σ (fast coupling)
        self.lorenz_rho     = CIFScalar(lambda: 28.0)         # Lorenz ρ (tipping threshold)
        self.ecn_inner      = CIFScalar(lambda: max(c.d_model // 2, 64))
        # ── loss weights ───────────────────────────────────────────────────
        self.hor_weight     = CIFScalar(lambda: 0.40)
        self.unc_weight     = CIFScalar(lambda: 0.05)
        self.act_weight     = CIFScalar(lambda: c.act_tau)
        self.lm_weight      = CIFScalar(lambda: 1.0)
        self.lyap_weight    = CIFScalar(lambda: 0.001)        # Lyapunov stability reg
        # ── training loop ──────────────────────────────────────────────────
        self.accum_steps    = CIFScalar(lambda: c.accum_steps)
        self.ckpt_interval  = CIFScalar(lambda: c.checkpoint_interval)
        self.max_steps_val  = CIFScalar(lambda: c.max_steps)
        # ── GQA Deny gates (boolean, flip-cached) ──────────────────────────
        # These wrap static checks computed once during model construction
        self._kv_multi_head = Deny(c)  # True if n_heads != n_kv_heads
        self._use_feedback  = Deny(c)  # True if fb_iterations > 0
        self._use_horizon   = Deny(c)  # True if max_horizon > 1

    # ── convenience boolean checks (Deny-locked after first call) ──────────
    def is_multi_head_kv(self) -> bool:
        """True when using GQA (n_heads > n_kv_heads). Deny-locked O(1)."""
        return self._kv_multi_head.test(lambda c: not (c.n_heads == c.n_kv_heads))

    def uses_feedback(self) -> bool:
        """True when FeedbackLoop iterations > 0. Deny-locked O(1)."""
        return self._use_feedback.test(lambda c: not (c.fb_iterations > 0))

    def uses_horizon(self) -> bool:
        """True when speculative horizon active. Deny-locked O(1)."""
        return self._use_horizon.test(lambda c: not (c.max_horizon > 1))


# ══════════════════════════════════════════════════════════════════════════════
# 3. PELL-LUCAS SPINE  [HST v1+2]  — Sequence Topology
# ══════════════════════════════════════════════════════════════════════════════

def build_spine_sequence(max_len: int) -> List[int]:
    """Pell-Lucas recurrence: P(n) = 2*P(n-1) + P(n-2)."""
    s = [0, 2, 6]
    while True:
        nxt = 2 * s[-1] + s[-2]
        if nxt >= max_len:
            break
        s.append(nxt)
    return s


class SpineAnalyzer:
    """
    Precomputes BFS graph structure for all spine nodes.
    Per-position BFS metadata is cached once at init — Deny-locked in
    CompleteLatticeProcessor for O(1) weight lookups.
    """
    def __init__(self, max_seq_len: int):
        self.spine     = build_spine_sequence(max_seq_len)
        self.spine_set = set(self.spine)
        self.spine_idx = {v: i for i, v in enumerate(self.spine)}
        self._cache: Dict[int, dict] = {}
        # CIF: pre-resolved spine membership for position 0..max_seq_len
        self._pos_is_spine: Dict[int, Deny] = {}
        logger.info(
            f"Pell-Lucas Spine | max_len={max_seq_len} | "
            f"nodes={len(self.spine)} | seq={self.spine}"
        )
        self._precompute()
        self._build_spine_deny_map(max_seq_len)

    def _get_ancestors(self, pos: int) -> List[int]:
        if pos not in self.spine_idx:
            prev = [s for s in self.spine if s < pos]
            return [prev[-1]] if prev else []
        idx = self.spine_idx[pos]
        return [self.spine[i] for i in [idx - 1, idx - 2, idx - 3] if i >= 0]

    def _precompute(self) -> None:
        for pos in self.spine:
            levels: Dict[int, List[int]] = {0: [pos]}
            visited = {pos}
            bfs = deque([(pos, 0)])
            path_counts: Dict[int, int] = {pos: 1}
            while bfs:
                curr, lvl = bfs.popleft()
                if lvl >= 6:
                    continue
                for a in self._get_ancestors(curr):
                    if lvl + 1 not in levels:
                        levels[lvl + 1] = []
                    if a not in levels[lvl + 1]:
                        levels[lvl + 1].append(a)
                    path_counts[a] = path_counts.get(a, 0) + path_counts[curr]
                    if a not in visited:
                        visited.add(a)
                        bfs.append((a, lvl + 1))
            self._cache[pos] = {
                "levels": levels,
                "path_counts": path_counts,
                "max_depth": max(levels.keys()) if levels else 0,
            }

    def _build_spine_deny_map(self, max_len: int) -> None:
        """
        [CIF.F] Pre-build Deny gates for every position up to max_len.
        Deny stores NOT(result); is_spine_pos returns `not cache` = True for spine.
        """
        spine_set = self.spine_set
        for pos in range(min(max_len, 2048)):
            d = Deny(pos)
            d.test(lambda p, s=spine_set: p in s)  # cache = not(p in spine_set)
            self._pos_is_spine[pos] = d

    def is_spine_pos(self, pos: int) -> bool:
        """O(1) Deny-locked spine membership test."""
        if pos in self._pos_is_spine:
            return not self._pos_is_spine[pos].cache  # cache holds NOT-of-result
        return pos in self.spine_set

    def get_structure(self, pos: int) -> dict:
        return self._cache.get(
            pos, self._cache.get(0, {"levels": {}, "path_counts": {}, "max_depth": 0})
        )


# ══════════════════════════════════════════════════════════════════════════════
# 4. GEOMETRIC MANIFOLDS  [HST v2]  — Embedding & Positional Encoding
# ══════════════════════════════════════════════════════════════════════════════

class HyperbolicEmbedding(nn.Module):
    """Poincaré ball projection — ||x|| < 1 for hierarchical representation."""
    def __init__(self, vocab_size: int, d_model: int, curvature: float = 1.0):
        super().__init__()
        self.c = curvature
        # CIF: max_norm is a static scalar — resolve once
        self._max_norm_cif = CIFScalar(lambda: (1.0 - 1e-5) / math.sqrt(curvature))
        self.emb = nn.Embedding(vocab_size, d_model)
        nn.init.normal_(self.emb.weight, mean=0.0, std=0.01)

    def _project(self, x: torch.Tensor) -> torch.Tensor:
        max_norm = self._max_norm_cif.get()           # O(1) CIF lookup
        norm = x.norm(dim=-1, keepdim=True).clamp(min=1e-9)
        return x * (max_norm / norm).clamp(max=1.0)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self._project(self.emb(ids))


class LatticePositionalEncoding(nn.Module):
    """Dual-stream: classical sinusoids + spine-proximity MLP. [HST v2]"""
    def __init__(self, d_model: int, max_seq_len: int):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
        t = torch.arange(max_seq_len).float()
        sinusoid = torch.einsum("i,j->ij", t, inv_freq)
        self.register_buffer("abs_pe", torch.cat([sinusoid.sin(), sinusoid.cos()], dim=-1))

        spine_list = build_spine_sequence(max_seq_len)
        self.register_buffer("spine", torch.tensor(spine_list, dtype=torch.float))
        # CIF: rel_net input dim is always 3
        self._rel_input_dim_cif = CIFScalar(lambda: 3)
        self.rel_net = nn.Sequential(
            nn.Linear(3, d_model // 4),
            nn.LayerNorm(d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, d_model),
        )

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        abs_emb = self.abs_pe[positions]
        sp = self.spine.to(positions.device)
        pos_f = positions.float().unsqueeze(-1)
        diffs = pos_f - sp
        ld = torch.where(diffs >= 0, diffs, torch.full_like(diffs, float("inf"))).min(dim=-1)[0].clamp(max=1e6)
        rd = torch.where(diffs < 0, -diffs, torch.full_like(diffs, float("inf"))).min(dim=-1)[0].clamp(max=1e6)
        rank = (diffs >= 0).sum(dim=-1).float()
        return abs_emb + self.rel_net(torch.stack([ld, rd, rank], dim=-1))


# ══════════════════════════════════════════════════════════════════════════════
# 5. ROTARY POSITION EMBEDDINGS (RoPE)  [HST v3]
# ══════════════════════════════════════════════════════════════════════════════

def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


class RotaryEmbedding(nn.Module):
    """RoPE — encodes relative position via rotation of Q and K. [HST v3]"""
    def __init__(self, head_dim: int, max_seq_len: int = 4096, base: int = 10000):
        super().__init__()
        # CIF: inv_freq scale is static
        self._half_dim_cif = CIFScalar(lambda: head_dim // 2)
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        t = torch.arange(seq_len, device=self.inv_freq.device).float()
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cache", emb.cos()[None, None, :, :])
        self.register_buffer("sin_cache", emb.sin()[None, None, :, :])

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, offset: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        S = q.size(2)
        cos = self.cos_cache[:, :, offset : offset + S, :].to(q.dtype)
        sin = self.sin_cache[:, :, offset : offset + S, :].to(q.dtype)
        return (q * cos) + (_rotate_half(q) * sin), (k * cos) + (_rotate_half(k) * sin)


# ══════════════════════════════════════════════════════════════════════════════
# 6. PAGED KV CACHE  [HST v1+v3  — CIF.H]
#    Uses Deny for allocation boundary detection — O(1) after first check.
# ══════════════════════════════════════════════════════════════════════════════

class PagedKVCache:
    """
    Block-level KV memory with full CIF coverage.
    [CIF.H] _alloc_cif: CachedCondition tracks allocation efficiency.
    """
    def __init__(self, n_kv_heads: int, head_dim: int, block_size: int, device: str):
        self.nh  = n_kv_heads
        self.hd  = head_dim
        self.bs  = block_size
        self.dev = device
        self.k_blocks: List[torch.Tensor] = []
        self.v_blocks: List[torch.Tensor] = []
        self.pos = 0
        # CIF.H: allocation boundary state
        self.alloc_state = CIFState(self, initial_deny=False)
        # CIF.A: track allocation call efficiency
        self._alloc_cif = CachedCondition(self, initial_deny=False)
        # CIF: block size locked
        self._block_size_cif = CIFScalar(lambda: block_size)

    def append(self, k: torch.Tensor, v: torch.Tensor) -> None:
        seq_len = k.size(2)
        needs_alloc = self.alloc_state.test(
            lambda s: not s.k_blocks or s.k_blocks[-1].size(2) >= s.bs
        )
        if needs_alloc:
            bs = self._block_size_cif.get()  # O(1)
            self.k_blocks.append(torch.zeros(1, self.nh, bs, self.hd, device=self.dev))
            self.v_blocks.append(torch.zeros(1, self.nh, bs, self.hd, device=self.dev))
            self.alloc_state.set_static()

        idx = self.pos % self.bs
        end = idx + seq_len
        if end <= self.bs:
            self.k_blocks[-1][:, :, idx:end, :] = k
            self.v_blocks[-1][:, :, idx:end, :] = v
        else:
            bs = self._block_size_cif.get()
            self.k_blocks.append(torch.zeros(1, self.nh, bs, self.hd, device=self.dev))
            self.v_blocks.append(torch.zeros(1, self.nh, bs, self.hd, device=self.dev))
            self.k_blocks[-1][:, :, :seq_len, :] = k
            self.v_blocks[-1][:, :, :seq_len, :] = v

        self.pos += seq_len
        if self.pos % self.bs == 0:
            self.alloc_state.set_dynamic()

    def get_context(self) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if not self.k_blocks:
            return None, None
        k = torch.cat(self.k_blocks, dim=2)[:, :, : self.pos, :]
        v = torch.cat(self.v_blocks, dim=2)[:, :, : self.pos, :]
        return k, v


# ══════════════════════════════════════════════════════════════════════════════
# 7. QK NORMALIZATION  [HST v3]  — Attention Stability
# ══════════════════════════════════════════════════════════════════════════════

class QKNorm(nn.Module):
    """Per-head L2 normalization on Q and K. [HST v3 + CIF.E]"""
    def __init__(self, head_dim: int):
        super().__init__()
        self.scale_q = nn.Parameter(torch.ones(head_dim))
        self.scale_k = nn.Parameter(torch.ones(head_dim))
        # CIF.E: the head_dim is static — locked via Deny
        self._head_dim_deny = Deny(head_dim)
        self._head_dim_deny.test(lambda d: not (d > 0))  # pre-evaluate once

    def forward(
        self, q: torch.Tensor, k: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return F.normalize(q, dim=-1) * self.scale_q, F.normalize(k, dim=-1) * self.scale_k


# ══════════════════════════════════════════════════════════════════════════════
# 8. GROUPED QUERY ATTENTION + RoPE + QK-NORM  [HST v3 + CIF.E]
# ══════════════════════════════════════════════════════════════════════════════

class GQAFlashAttention(nn.Module):
    """
    GQA + RoPE + QK-Norm with full CIF coverage. [HST v3]
    CIF.E: kv_ratio, attn_scale, head_dim all Deny-locked after first call.
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        cif: CIFRegistry,
        rope: RotaryEmbedding,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.nh   = n_heads
        self.nkv  = n_kv_heads
        self.hd   = cif.head_dim.get()       # O(1)
        self.cif  = cif
        self.rope = rope

        # CIF.E: kv_ratio and scale are static — Deny-lock them
        self._ratio_cif  = CIFScalar(lambda: n_heads // n_kv_heads)
        self._scale_deny = Deny(cif)
        self._scale_deny.test(lambda c: not (c.attn_scale.get() > 0))  # pre-eval

        self.q_proj   = nn.Linear(d_model, n_heads   * self.hd, bias=False)
        self.k_proj   = nn.Linear(d_model, n_kv_heads * self.hd, bias=False)
        self.v_proj   = nn.Linear(d_model, n_kv_heads * self.hd, bias=False)
        self.out_proj = nn.Linear(d_model,  d_model,  bias=False)
        self.qk_norm  = QKNorm(self.hd)
        self.drop_p   = dropout

    def forward(
        self,
        x: torch.Tensor,
        cache: Optional[PagedKVCache] = None,
    ) -> Tuple[torch.Tensor, Optional[PagedKVCache]]:
        B, S, D = x.shape
        offset = cache.pos if cache is not None else 0
        ratio  = self._ratio_cif.get()   # O(1)

        q = self.q_proj(x).view(B, S, self.nh,  self.hd).permute(0, 2, 1, 3)
        k = self.k_proj(x).view(B, S, self.nkv, self.hd).permute(0, 2, 1, 3)
        v = self.v_proj(x).view(B, S, self.nkv, self.hd).permute(0, 2, 1, 3)

        # RoPE + QK-Norm
        q, k = self.rope(q, k, offset=offset)
        q, k = self.qk_norm(q, k)

        if cache is not None:
            cache.append(k, v)
            ck, cv = cache.get_context()
            if ck is not None:
                k, v = ck.expand(B, -1, -1, -1), cv.expand(B, -1, -1, -1)

        # GQA expand — ratio from CIFScalar O(1)
        k = k.repeat_interleave(ratio, dim=1)
        v = v.repeat_interleave(ratio, dim=1)

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.drop_p if self.training else 0.0,
            is_causal=(cache is None),
        ).transpose(1, 2).contiguous().view(B, S, D)

        return self.out_proj(out), cache


# ══════════════════════════════════════════════════════════════════════════════
# 9. DIAMOND MIXER  [HST v1]  — Lossless Logic FFN (even layers)
# ══════════════════════════════════════════════════════════════════════════════

class DiamondMixer(nn.Module):
    """F(a,b) = GELU(a+b)*tanh(a-b) + SiLU(b). [HST v1]"""
    def __init__(self, d_model: int, cif: CIFRegistry):
        super().__init__()
        inner = cif.d_model_x2.get()   # O(1)
        self.norm = nn.LayerNorm(d_model)
        self.up   = nn.Linear(d_model, inner, bias=False)
        self.down = nn.Linear(inner,   d_model, bias=False)
        # CIF: inner dim is static
        self._inner_cif = CIFScalar(lambda: inner)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = self.up(self.norm(x))
        a, b = u.chunk(2, dim=-1)
        return x + self.down(torch.cat([F.gelu(a + b) * torch.tanh(a - b), F.silu(b)], dim=-1))


# ══════════════════════════════════════════════════════════════════════════════
# 10. SWIGLU FFN  [HST v3]  — Gated Linear Unit (odd layers)
# ══════════════════════════════════════════════════════════════════════════════

class SwiGLUFFN(nn.Module):
    """SiLU(x·W_gate) ⊙ (x·W_up) · W_down. [HST v3]"""
    def __init__(self, d_model: int, cif: CIFRegistry):
        super().__init__()
        inner = cif.swiglu_inner.get()   # O(1)
        self.norm      = nn.LayerNorm(d_model)
        self.gate_proj = nn.Linear(d_model, inner, bias=False)
        self.up_proj   = nn.Linear(d_model, inner, bias=False)
        self.down_proj = nn.Linear(inner,   d_model, bias=False)
        # CIF: inner dim static
        self._inner_cif = CIFScalar(lambda: inner)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xn = self.norm(x)
        return x + self.down_proj(F.silu(self.gate_proj(xn)) * self.up_proj(xn))


# ══════════════════════════════════════════════════════════════════════════════
# 11. FEEDBACK LOOP  [HST v2]  — GRU Iterative Self-Correction
# ══════════════════════════════════════════════════════════════════════════════

class FeedbackLoop(nn.Module):
    """GRU-based iterative hidden-state refinement. [HST v2 + CIF]"""
    def __init__(self, d_model: int, cif: CIFRegistry):
        super().__init__()
        self.cif  = cif
        self.gru  = nn.GRUCell(d_model, d_model)
        self.gate = nn.Linear(d_model, 1)
        # CIF: number of iterations is static after config
        self._iters_deny = Deny(cif)
        self._iters_deny.test(lambda c: not (c.fb_iters.get() > 0))  # pre-eval
        # CIF.A: track gate computation efficiency
        self._gate_cif = CachedCondition(cif, initial_deny=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        h = x.reshape(-1, D)
        iters = self.cif.fb_iters.get()   # O(1)
        for _ in range(iters):
            g = torch.sigmoid(self.gate(h))
            h = (1 - g) * h + g * self.gru(h, h)
        return h.view(B, S, D)


# ══════════════════════════════════════════════════════════════════════════════
# 12. HEBBIAN FAST WEIGHTS  [HST v2 + CIF.D]
#     decay constant Deny-locked: O(1) float lookup on every forward
# ══════════════════════════════════════════════════════════════════════════════

class HebbianFastWeights(nn.Module):
    """
    Inference-time associative memory.  [HST v2]
    [CIF.D] _decay_deny: the decay constant is locked via Deny after first call.
    """
    def __init__(self, d_model: int, cif: CIFRegistry):
        super().__init__()
        self.cif    = cif
        d_small     = cif.hebb_d_small.get()   # O(1)  192 for d_model=768
        d_half      = cif.hebb_d_half.get()    # O(1)  384 for d_model=768
        # Shapes match the original checkpoint exactly:
        #   q_proj  : [d_model -> d_small]
        #   kv_proj : [d_model -> d_half*2]  (k and v each d_half)
        #   out_proj: [d_small -> d_model]
        # The forward contracts over the sequence dimension (s), so the
        # associative memory is [B, d_half, d_half].  A learned down-projection
        # squeezes it to d_small before out_proj, keeping all shapes consistent.
        self.q_proj   = nn.Linear(d_model, d_small,  bias=False)
        self.kv_proj  = nn.Linear(d_model, d_half * 2, bias=False)
        self.mem_proj = nn.Linear(d_half,  d_small,  bias=False)  # new: squeeze mem
        self.out_proj = nn.Linear(d_small, d_model,  bias=False)
        self._decay_deny = Deny(cif)
        self._decay_deny.test(lambda c: not (c.hebb_decay.get() > 0))
        self._decay_val  = cif.hebb_decay   # CIFScalar → O(1).get()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        kv = self.kv_proj(x)                            # [B, S, d_half*2]
        k, v = kv.chunk(2, dim=-1)                      # each [B, S, d_half]
        decay = self._decay_val.get()                   # O(1)
        # Associative memory: outer product over sequence → [B, d_half, d_half]
        associative_mem = torch.einsum("bsi,bsj->bij", v, k) * decay
        query = self.q_proj(x)                          # [B, S, d_small]
        # Squeeze memory to d_small so query can contract over it
        mem_small = self.mem_proj(associative_mem)      # [B, d_half, d_small]
        # retrieved[b,s,e] = sum_i query[b,s,i] * mem_small[b,?,i]
        # mem_small is [B, d_half, d_small]; we want [B, d_small, d_small]
        # → take mean over d_half dimension (pool context)
        mem_pooled = mem_small.mean(dim=1)              # [B, d_small]
        retrieved  = query * mem_pooled.unsqueeze(1)    # [B, S, d_small]
        return x + self.out_proj(retrieved)


# ══════════════════════════════════════════════════════════════════════════════
# 13. COMPLETE LATTICE PROCESSOR  [HST v2 + CIF.F]
#     Per-spine-position BFS weights Deny-locked → O(1) on every forward
# ══════════════════════════════════════════════════════════════════════════════

class CompleteLatticeProcessor(nn.Module):
    """
    BFS path-weighted Pell-Lucas message passing. [HST v2]
    [CIF.F] Per-spine-position softmax weights are Deny-locked after first
    forward pass — subsequent calls are O(1) tensor lookups.
    """
    def __init__(self, d_model: int, analyzer: SpineAnalyzer, cif: CIFRegistry):
        super().__init__()
        self.analyzer = analyzer
        self.fuse = nn.Sequential(
            nn.Linear(cif.d_model_x2.get(), d_model),  # O(1)
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        self._spine_list = analyzer.spine
        # CIF.F: weight tensors for each spine position — computed once, reused
        self._weight_cache: Dict[int, torch.Tensor] = {}
        # CIF.F: Deny flags per spine position "has ancestors been resolved?"
        self._resolved: Dict[int, Deny] = {}

    def _get_weights(self, pos: int, device: torch.device) -> Optional[torch.Tensor]:
        """Return Deny-locked softmax weights for spine position `pos`."""
        if pos not in self._weight_cache:
            meta = self.analyzer.get_structure(pos)
            ancestors = meta["levels"].get(1, [])
            if not ancestors:
                return None
            raw = torch.tensor(
                [float(meta["path_counts"].get(a, 1)) for a in ancestors],
                dtype=torch.float32,
            )
            self._weight_cache[pos] = F.softmax(raw, dim=0)
            # Deny-lock: this position's weights will never change
            d = Deny(pos)
            d.test(lambda p: not (p in self._weight_cache))  # pre-evaluate → False
            self._resolved[pos] = d
        return self._weight_cache[pos].to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        updates = torch.zeros_like(x)
        for pos in self._spine_list:
            if pos >= S:
                break
            meta  = self.analyzer.get_structure(pos)
            ancestors = [a for a in meta["levels"].get(1, []) if a < S]
            if not ancestors:
                continue
            weights = self._get_weights(pos, x.device)  # O(1) after first call
            if weights is None:
                continue
            # Trim weights if some ancestors are out of range
            w = weights[: len(ancestors)]
            if w.sum() > 0:
                w = w / w.sum()
            states = torch.stack([x[:, a, :] for a in ancestors], dim=1)
            agg = (states * w.view(1, -1, 1)).sum(dim=1)
            updates[:, pos, :] = self.fuse(torch.cat([x[:, pos, :], agg], dim=-1))
        return x + updates


# ══════════════════════════════════════════════════════════════════════════════
# 14. HYPER-LATTICE AGGREGATOR  [HST v3 + CIF.E]
#     Attention scale is Deny-locked; k% of paths is CIF-cached [v8 ref]
# ══════════════════════════════════════════════════════════════════════════════

class HyperLatticeAggregator(nn.Module):
    """
    Cross-layer hidden-state fusion via learned attention. [HST v3]
    [CIF.E] _scale_cif: attention scale locked after first call.
    [CIF — v8 ref pattern] _path_state Deny for k-path selection.
    """
    def __init__(self, d_model: int, compress_dim: int = 128):
        super().__init__()
        self.q_proj = nn.Linear(d_model, compress_dim, bias=False)
        self.k_proj = nn.Linear(d_model, compress_dim, bias=False)
        self.v_proj = nn.Linear(d_model, d_model,      bias=False)
        self.out    = nn.Linear(d_model, d_model,      bias=False)
        self.norm   = nn.LayerNorm(d_model)
        self.gate   = nn.Parameter(torch.zeros(1))
        # CIF.E: attention scale is static
        self._scale_cif  = CIFScalar(lambda: 1.0 / math.sqrt(compress_dim))
        # CIF (v8 ref): path k% is static — Deny-lock it
        self._path_deny  = Deny(compress_dim)
        self._path_deny.test(lambda d: not (d > 0))  # pre-eval

    def forward(self, x: torch.Tensor, layer_states: List[torch.Tensor]) -> torch.Tensor:
        if not layer_states:
            return x
        B, S, D = x.shape
        scale = self._scale_cif.get()          # O(1)
        pooled = torch.stack([ls.mean(dim=1) for ls in layer_states], dim=1)  # [B, n, D]
        xn = self.norm(x)
        q   = self.q_proj(xn.mean(dim=1, keepdim=True))   # [B, 1, C]
        k   = self.k_proj(pooled)                          # [B, n, C]
        v   = self.v_proj(pooled)                          # [B, n, D]
        attn = F.softmax((q @ k.transpose(-1, -2)) * scale, dim=-1)
        agg  = self.out(attn @ v).expand(-1, S, -1)        # [B, S, D]
        return x + torch.sigmoid(self.gate) * agg


# ══════════════════════════════════════════════════════════════════════════════
# 15. ADAPTIVE COMPUTATION TIME  [HST v3]  — Per-token Soft Halting
# ══════════════════════════════════════════════════════════════════════════════

class ACTHalter(nn.Module):
    """Halting probability p_h ∈ (0,1) per token. [HST v3]"""
    def __init__(self, d_model: int):
        super().__init__()
        self.halt_proj = nn.Linear(d_model, 1)
        # CIF: the projection output is always scalar — lock via Deny
        self._out_dim_deny = Deny(1)
        self._out_dim_deny.test(lambda d: not (d == 1))  # pre-eval

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.halt_proj(x)).squeeze(-1)


# ══════════════════════════════════════════════════════════════════════════════
# 16. SPECULATIVE HORIZON  [HST v1+2]  — Weight-tied multi-step prediction
# ══════════════════════════════════════════════════════════════════════════════

class SpeculativeHorizon(nn.Module):
    """Multi-step future token drafting. [HST v1+2]"""
    def __init__(self, d_model: int, vocab_size: int, max_h: int):
        super().__init__()
        self.max_h = max_h
        # CIF: max_h and vocab_size are static
        self._max_h_cif   = CIFScalar(lambda: max_h)
        self._vocab_cif   = CIFScalar(lambda: vocab_size)
        self.unc = nn.Sequential(
            nn.Linear(d_model, max(d_model // 8, 16)), nn.GELU(),
            nn.Linear(max(d_model // 8, 16), 1), nn.Sigmoid(),
        )
        self.step_emb    = nn.Embedding(max_h, d_model)
        self.shared_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model, bias=False),
            nn.GELU(),
        )
        self.out_proj = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ht          = h[:, -1, :]
        uncertainty = self.unc(ht)
        h_len       = (self._max_h_cif.get() * (1.0 - uncertainty)).long().clamp(2, self._max_h_cif.get())
        steps       = torch.arange(self._max_h_cif.get(), device=h.device)
        conditioned = ht.unsqueeze(1) + self.step_emb(steps).unsqueeze(0)
        return self.out_proj(self.shared_proj(conditioned)), h_len, uncertainty


# ══════════════════════════════════════════════════════════════════════════════
# 17. CHAOS LOGIC LAYER  [HST Chaos v3.2]
#
#  Implements three coupled chaotic systems:
#   A) Logistic map:  x_{n+1} = r·x_n·(1 − x_n), r=3.9 → full chaos regime
#   B) Lorenz σ-ρ coupling:  σ(y−x) mixing step, one Euler dt=0.01
#   C) Lyapunov exponent estimator: scalar per-token stability measure
#      Used as a regularisation signal during training.
#
#  Placed after every HyperLattice trigger layer (3, 7, 11).
# ══════════════════════════════════════════════════════════════════════════════

class ChaosLogicLayer(nn.Module):
    """
    Sensitive-dependence non-linear mixing with stability estimation. [HST v3.2]

    Logistic map  (r=3.9, chaotic regime):
        h_sig = σ(W_in · LayerNorm(x))
        h_chaos = r · h_sig · (1 − h_sig)

    Lorenz σ-ρ step (dt=0.01, one Euler step):
        fast  = SiLU(W_σ · x̃)          -- σ-variable (fast timescale)
        slow  = tanh(W_ρ · x̃)          -- ρ-variable (slow timescale)
        lorenz_Δ = fast · (slow − fast) · dt

    Lyapunov estimator:
        λ̂ = W_lyap · x̃                 -- learned per-token exponent
        Positive → chaotic, negative → convergent.
        Penalised toward zero in training (stability regularisation).

    Output: x + sigmoid(gate) · (h_chaos · D^{-½} + W_out(lorenz_Δ))
    """
    def __init__(self, d_model: int, cif: CIFRegistry):
        super().__init__()
        d_half = cif.hebb_d_half.get()   # O(1)
        self.norm      = nn.LayerNorm(d_model)
        # Logistic map projection (input → sigmoid space)
        self.chaos_in  = nn.Linear(d_model, d_model, bias=False)
        # Lorenz σ-ρ coupling projections
        self.sigma_proj = nn.Linear(d_model, d_half, bias=False)
        self.rho_proj   = nn.Linear(d_model, d_half, bias=False)
        self.lorenz_out = nn.Linear(d_half,  d_model, bias=False)
        # Lyapunov exponent estimator
        self.lyapunov   = nn.Linear(d_model, 1)
        # Learnable residual gate (starts closed — chaos gradually opens)
        self.gate       = nn.Parameter(torch.tensor(-2.0))

        # CIF: chaos params are static scalars — lock all at init
        self._r_cif     = cif.chaos_r    # CIFScalar → 3.9
        self._dt_cif    = cif.chaos_dt   # CIFScalar → 0.01
        # CIF: Deny-lock the "is r in chaotic regime?" check
        self._chaos_regime_deny = Deny(cif)
        self._chaos_regime_deny.test(lambda c: not (c.chaos_r.get() > 3.57))
        # CIF: d_model scale factor locked
        self._scale_cif = CIFScalar(lambda: d_model ** -0.5)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        out       : torch.Tensor  [B, S, D]  chaos-mixed hidden states
        lyap_est  : torch.Tensor  [B, S]     Lyapunov exponent estimate
        """
        xn  = self.norm(x)
        r   = self._r_cif.get()         # O(1) CIFScalar
        dt  = self._dt_cif.get()        # O(1) CIFScalar
        sc  = self._scale_cif.get()     # O(1) CIFScalar

        # ── A: Logistic map ────────────────────────────────────────────────
        h   = torch.sigmoid(self.chaos_in(xn))
        h_chaos = r * h * (1.0 - h)         # [B, S, D]

        # ── B: Lorenz σ-ρ one-step coupling ───────────────────────────────
        fast  = F.silu(self.sigma_proj(xn))  # σ-variable (fast)
        slow  = torch.tanh(self.rho_proj(xn))  # ρ-variable (slow)
        lorenz_delta = fast * (slow - fast) * dt   # one Euler step
        lorenz_out   = self.lorenz_out(lorenz_delta)   # [B, S, D]

        # ── C: Lyapunov exponent estimate ─────────────────────────────────
        lyap_est = self.lyapunov(xn).squeeze(-1)        # [B, S]

        # ── Gated residual ────────────────────────────────────────────────
        g   = torch.sigmoid(self.gate)
        chaos_signal = h_chaos * sc + lorenz_out       # combine
        out = x + g * chaos_signal

        return out, lyap_est


# ══════════════════════════════════════════════════════════════════════════════
# 18. ERROR CORRECTION NETWORK  [HST Error Nets v3.2]
#
#  A three-path network that learns the systematic residual error made by
#  the main transformer stack and generates an additive correction in logit
#  space.  It self-trains through the main LM cross-entropy objective — no
#  separate loss needed.  The correction gate starts near 0 and opens as
#  the network learns to improve predictions.
#
#  Architecture:
#   A) Error Detector   — per-token error magnitude in [0,1]
#   B) Error Direction  — correction direction in hidden space
#   C) Correction Head  — vocab-space delta (separate weights from lm_head)
# ══════════════════════════════════════════════════════════════════════════════

class ErrorCorrectionNetwork(nn.Module):
    """
    Learns and corrects the main model's systematic prediction errors. [HST v3.2]

    final_logits = lm_head(h) + sigmoid(gate) · correction_head(dir · mag)

    The ECN receives the same normalised hidden states as lm_head.
    During training the combined logits are used for cross-entropy,
    so the ECN gradient signal comes entirely from the LM loss.
    """
    def __init__(self, d_model: int, vocab_size: int, cif: CIFRegistry):
        super().__init__()
        d_inner = cif.ecn_inner.get()    # O(1)
        self.norm = nn.LayerNorm(d_model)

        # A. Error magnitude detector
        self.err_proj   = nn.Linear(d_model, d_inner, bias=False)
        self.err_mag    = nn.Sequential(
            nn.Linear(d_inner, max(d_inner // 4, 16)),
            nn.GELU(),
            nn.Linear(max(d_inner // 4, 16), 1),
            nn.Sigmoid(),
        )

        # B. Error direction extractor
        self.err_dir    = nn.Linear(d_model, d_inner, bias=False)

        # C. Correction head (separate from weight-tied lm_head)
        self.correction = nn.Linear(d_inner, vocab_size, bias=False)

        # Starts at sigmoid(-2) ≈ 0.12 → near-zero correction early in training
        self.gate       = nn.Parameter(torch.tensor(-2.0))

        # CIF: static dims locked
        self._d_inner_cif = CIFScalar(lambda: d_inner)
        self._vocab_cif   = CIFScalar(lambda: vocab_size)
        # CIF.A: track how many tokens get a meaningful correction
        self._corr_cif  = CachedCondition(self, initial_deny=False)

    def forward(
        self, h: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        h  : torch.Tensor  [B, S, D]  normalised hidden states

        Returns
        -------
        correction  : torch.Tensor  [B, S, V]  additive logit correction
        error_mag   : torch.Tensor  [B, S]     per-token error magnitude
        """
        hn = self.norm(h)

        # A: error magnitude
        err_feat = F.gelu(self.err_proj(hn))
        err_mag  = self.err_mag(err_feat).squeeze(-1)      # [B, S]

        # B: error direction
        err_dir  = F.gelu(self.err_dir(hn))                # [B, S, d_inner]

        # C: correction scaled by magnitude
        correction = self.correction(err_dir * err_mag.unsqueeze(-1))  # [B, S, V]

        g = torch.sigmoid(self.gate)
        return correction * g, err_mag


# ══════════════════════════════════════════════════════════════════════════════
# 19. ADAPTIVE TRANSFORMER BLOCK v3.2  [CIF.C — per-layer Deny flags]
#     _is_diamond_cif: Deny  → FFN parity locked at init, O(1) every forward
#     _thresh_cif: CIFScalar → exit threshold locked at init
# ══════════════════════════════════════════════════════════════════════════════

class AdaptiveBlock(nn.Module):
    """
    One transformer layer — all static decisions Deny-locked. [all HST versions]
    [CIF.C] _is_diamond_cif: Deny, layer FFN type locked permanently.
    [CIF.C] _thresh_cif: exit threshold locked permanently.
    """
    def __init__(
        self,
        layer_idx: int,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        cif: CIFRegistry,
        rope: RotaryEmbedding,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.cif       = cif
        self.layer_idx = layer_idx
        self.norm1     = nn.LayerNorm(d_model)
        self.attn      = GQAFlashAttention(d_model, n_heads, n_kv_heads, cif, rope, dropout)

        # CIF.C: FFN type locked via Deny — O(1) bool after first call
        _is_diamond_at_init = (layer_idx % 2 == 0)
        self._is_diamond_deny = Deny(layer_idx)
        # Pre-evaluate: caches NOT(even) = True for odd, False for even
        self._is_diamond_deny.test(lambda i: not (i % 2 == 0))
        # Determine correct FFN from locked value
        use_diamond = _is_diamond_at_init
        self.ffn: nn.Module = DiamondMixer(d_model, cif) if use_diamond else SwiGLUFFN(d_model, cif)
        self.ffn_name = "Diamond" if use_diamond else "SwiGLU"

        self.drop      = nn.Dropout(dropout)
        self.act_halt  = ACTHalter(d_model)

        # CIF.C: exit threshold locked after first get()
        self._thresh_cif = cif.exit_thresh  # CIFScalar reference

        self.conf_gate = nn.Sequential(nn.Linear(d_model, 1), nn.Sigmoid())

    def forward(
        self,
        x: torch.Tensor,
        cache: Optional[PagedKVCache] = None,
    ) -> Tuple[torch.Tensor, Optional[PagedKVCache], bool, torch.Tensor, torch.Tensor]:
        a_res, next_cache = self.attn(self.norm1(x), cache)
        x   = x + self.drop(a_res)
        x   = self.ffn(x)
        conf = self.conf_gate(x[:, -1, :]).mean()
        # CIF.C: threshold O(1)
        stop = bool(conf.item() > self._thresh_cif.get())
        halt_prob = self.act_halt(x)
        return x, next_cache, stop, conf, halt_prob


# ══════════════════════════════════════════════════════════════════════════════
# 20. IO v3.2 MODEL — Full Integration with CIF + Chaos + Error Networks
# ══════════════════════════════════════════════════════════════════════════════

class Io(nn.Module):
    """
    Io v3.2 — HST model with Chaos Logic and Error Correction Networks.
    [CIF.C] HyperLattice trigger set pre-resolved to frozenset → O(1) check.
    [Chaos]  ChaosLogicLayer applied at each HyperLattice trigger layer.
    [ECN]    ErrorCorrectionNetwork adds learned correction to final logits.
    """
    def __init__(self, cfg: IoConfig):
        super().__init__()
        self.cfg      = cfg
        self.cif      = CIFRegistry(cfg)
        self.analyzer = SpineAnalyzer(cfg.max_seq_len)

        # Shared RoPE
        self.rope = RotaryEmbedding(self.cif.head_dim.get(), cfg.max_seq_len)

        # Input pipeline
        self.embedding = HyperbolicEmbedding(cfg.vocab_size, cfg.d_model)
        self.pos_enc   = LatticePositionalEncoding(cfg.d_model, cfg.max_seq_len)
        self.drop      = nn.Dropout(cfg.dropout)

        # Lattice
        self.lattice = CompleteLatticeProcessor(cfg.d_model, self.analyzer, self.cif)

        # Transformer stack
        self.layers = nn.ModuleList([
            AdaptiveBlock(i, cfg.d_model, cfg.n_heads, cfg.n_kv_heads,
                          self.cif, self.rope, cfg.dropout)
            for i in range(cfg.n_layers)
        ])

        # Hyper-Lattice modules
        n_hyper = self.cif.n_hyper.get()   # O(1)
        compress = self.cif.hyper_compress.get()  # O(1)
        self.hyper_lattices = nn.ModuleList([
            HyperLatticeAggregator(cfg.d_model, compress)
            for _ in range(n_hyper)
        ])
        self.hyper_interval = cfg.hyper_interval

        # ── v3.2: Chaos Logic Layers (one per HyperLattice trigger) ───────
        self.chaos_layers = nn.ModuleList([
            ChaosLogicLayer(cfg.d_model, self.cif)
            for _ in range(n_hyper)
        ])

        # CIF.C: HyperLattice trigger layer indices locked as frozenset
        self._hyper_triggers: FrozenSet[int] = self.cif.hyper_triggers.get()  # O(1)
        # Deny-lock the trigger check for each layer.
        # Deny stores NOT(result), so `not cache` = True iff layer is a trigger.
        self._layer_is_hyper: Dict[int, Deny] = {}
        trig = self._hyper_triggers
        for i in range(cfg.n_layers):
            d = Deny(i)
            d.test(lambda idx, t=trig: idx in t)  # cache = not(idx in triggers)
            self._layer_is_hyper[i] = d

        # Post-stack refiners
        self.feedback = FeedbackLoop(cfg.d_model, self.cif)
        self.hebbian  = HebbianFastWeights(cfg.d_model, self.cif)

        # Output heads
        self.norm_out = nn.LayerNorm(cfg.d_model)
        self.lm_head  = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.horizon  = SpeculativeHorizon(cfg.d_model, cfg.vocab_size, cfg.max_horizon)

        # ── v3.2: Error Correction Network ────────────────────────────────
        self.error_net = ErrorCorrectionNetwork(cfg.d_model, cfg.vocab_size, self.cif)

        # Weight tying (lm_head + horizon only — ECN correction head is intentionally separate)
        self.lm_head.weight = self.embedding.emb.weight
        self.horizon.out_proj.weight = self.embedding.emb.weight

        # CIF.G: training-loop booleans locked at construction
        self._uses_feedback_deny = Deny(cfg)
        self._uses_feedback_deny.test(lambda c: not (c.fb_iterations > 0))

        self.apply(self._init_weights)

        n_params = sum(p.numel() for p in self.parameters())
        size_mb  = n_params * 4 / 1024 / 1024
        logger.info(
            f"Io v3.2 | Layers={cfg.n_layers} | d_model={cfg.d_model} | "
            f"n_heads={cfg.n_heads} | n_kv_heads={cfg.n_kv_heads} | "
            f"Params={n_params:,} | Size≈{size_mb:.1f}MB (float32)"
        )
        ffn_pat = " ".join("D" if i % 2 == 0 else "S" for i in range(cfg.n_layers))
        logger.info(f"FFN pattern (D=Diamond, S=SwiGLU): [{ffn_pat}]")
        logger.info(f"HyperLattice triggers (Deny-locked): {sorted(self._hyper_triggers)}")
        logger.info(f"ChaosLogicLayers: {n_hyper} (one per HyperLattice trigger, r={self.cif.chaos_r.get()})")
        logger.info(f"ErrorCorrectionNetwork: d_inner={self.cif.ecn_inner.get()}, gate_init=sigmoid(-2)≈0.12")
        logger.info(f"CIF coverage: hyper_triggers=frozenset, {n_params:,} params all CIF-init")

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def _make_caches(self) -> List[PagedKVCache]:
        bs = self.cif.block_size.get()   # O(1)
        hd = self.cif.head_dim.get()     # O(1)
        return [
            PagedKVCache(self.cfg.n_kv_heads, hd, bs, self.cfg.device)
            for _ in range(self.cfg.n_layers)
        ]

    def forward(
        self,
        ids: torch.Tensor,
        caches: Optional[List[PagedKVCache]] = None,
        training: bool = True,
    ) -> Dict[str, Any]:
        B, S = ids.shape

        x = self.embedding(ids)
        p_offset  = caches[0].pos if caches else 0
        positions = torch.arange(p_offset, p_offset + S, device=ids.device).unsqueeze(0).expand(B, -1)
        x = self.drop(x + self.pos_enc(positions))
        x = self.lattice(x)

        new_caches: List[Optional[PagedKVCache]] = []
        layer_states: List[torch.Tensor] = []
        hyper_idx = 0
        all_halt_probs: List[torch.Tensor] = []
        lyap_estimates: List[torch.Tensor] = []   # v3.2: Lyapunov estimates from chaos layers

        for i, layer in enumerate(self.layers):
            cache_i = caches[i] if caches else None
            x, nc, stop, _, halt_prob = layer(x, cache_i)
            new_caches.append(nc)
            layer_states.append(x)
            all_halt_probs.append(halt_prob)

            # CIF.C: O(1) Deny-locked HyperLattice trigger check
            is_hyper = not self._layer_is_hyper[i].cache  # cache holds NOT-of-result
            if is_hyper:
                # v3.1: cross-layer aggregation
                x = self.hyper_lattices[hyper_idx](x, layer_states[: i + 1])
                # v3.2: chaos mixing after HyperLattice
                x, lyap = self.chaos_layers[hyper_idx](x)
                lyap_estimates.append(lyap)
                hyper_idx += 1

            if not training and stop and S == 1:
                new_caches.extend(
                    caches[i + 1 :] if caches else [None] * (self.cfg.n_layers - i - 1)
                )
                pad = torch.zeros_like(halt_prob)
                all_halt_probs.extend([pad] * (self.cfg.n_layers - i - 1))
                break

        act_cost = torch.stack([h.mean() for h in all_halt_probs]).sum()

        # v3.2: Lyapunov stability cost — penalise divergent chaos
        if lyap_estimates:
            lyap_cost = torch.stack([le.mean() for le in lyap_estimates]).mean().abs()
        else:
            lyap_cost = torch.zeros(1, device=ids.device)

        x = self.feedback(x)
        if not training:
            x = self.hebbian(x)

        h      = self.norm_out(x)
        logits = self.lm_head(h)

        # v3.2: ECN additive correction — folds into logits; ECN trains via LM loss
        ecn_correction, error_mag = self.error_net(h)
        logits = logits + ecn_correction

        drafts, h_len, unc = self.horizon(h)

        return {
            "logits":     logits,
            "drafts":     drafts,
            "h_len":      h_len,
            "unc":        unc,
            "act_cost":   act_cost,
            "lyap_cost":  lyap_cost,       # v3.2
            "error_mag":  error_mag,       # v3.2 (monitoring only)
            "caches":     new_caches,
        }


# ══════════════════════════════════════════════════════════════════════════════
# 19. Q2AC2D-17 QUALITY SCORER  [HST v3]
# ══════════════════════════════════════════════════════════════════════════════

def q2ac2d17_quality(ids: torch.Tensor) -> torch.Tensor:
    """Quality-Aware Adaptive Computation Control — 17 heuristic dimensions."""
    B, S = ids.shape
    weights: List[float] = []
    for b in range(B):
        seq = ids[b]
        uniq, counts = torch.unique(seq, return_counts=True)
        probs     = counts.float() / S
        entropy   = -(probs * probs.clamp(min=1e-9).log()).sum().item()
        ent_norm  = entropy / math.log(max(len(counts), 2))
        diversity = len(uniq) / S
        if S > 1:
            bg    = torch.stack([seq[:-1], seq[1:]], dim=1)
            bg_div = len(torch.unique(bg, dim=0)) / (S - 1)
        else:
            bg_div = 0.0
        if S > 2:
            tg  = [(seq[j].item(), seq[j+1].item(), seq[j+2].item()) for j in range(S - 2)]
            rep = len(set(tg)) / len(tg)
        else:
            rep = 1.0
        len_score = 1.0 - abs(S - 384) / max(S, 384)
        score = 0.35*ent_norm + 0.25*diversity + 0.20*bg_div + 0.12*rep + 0.08*len_score
        weights.append(0.5 + 1.5 * score)
    return torch.tensor(weights, dtype=torch.float32, device=ids.device)


# ══════════════════════════════════════════════════════════════════════════════
# 20. PELL-LUCAS POSITION-WEIGHTED LOSS  [HST v3 + CIF.F]
#     Per-position spine-membership uses SpineAnalyzer.is_spine_pos()
#     which is Deny-locked → O(1) per position after first call
# ══════════════════════════════════════════════════════════════════════════════

_pl_weight_cache: Dict[int, torch.Tensor] = {}

def pell_lucas_loss_weights(
    seq_len: int, device: torch.device, analyzer: SpineAnalyzer
) -> torch.Tensor:
    """
    Spine-anchor position-weighted loss.
    [CIF.F] Uses Deny-locked spine membership tests — O(1) after first eval.
    """
    global _pl_weight_cache
    if seq_len in _pl_weight_cache:
        return _pl_weight_cache[seq_len].to(device)

    spine_list = analyzer.spine
    w: List[float] = []
    for i in range(seq_len):
        # CIF.F: O(1) Deny-locked membership test
        if analyzer.is_spine_pos(i):
            w.append(2.0)
        else:
            dists = [abs(i - s) for s in spine_list if s <= i + 50]
            min_dist = min(dists) if dists else seq_len
            w.append(1.0 + 1.0 / (1.0 + min_dist))

    t = torch.tensor(w, dtype=torch.float32)
    _pl_weight_cache[seq_len] = t
    return t.to(device)


# ══════════════════════════════════════════════════════════════════════════════
# SELF-IDENTITY CORPUS  [HST v3]  — Io knows who and what it is
# ══════════════════════════════════════════════════════════════════════════════

SELF_IDENTITY_CORPUS: List[str] = [
    "I am Io. I am a language model built on the High-Speculative Topology (HST) v3.1 architecture.",
    "My name is Io. I was designed as a hybrid intelligence engine combining learned language generation with live web search and code execution.",
    "I am Io, a neural language model with approximately 100 million parameters trained on high-quality educational text.",
    "My architecture is HST v3.1, which stands for High-Speculative Topology. It combines geometric, algebraic, and probabilistic methods.",
    "I use Grouped Query Attention with Rotary Position Embeddings (RoPE) for efficient and accurate attention computation.",
    "My feed-forward layers alternate between the Diamond Mixer and SwiGLU activations, giving me a hybrid computation pathway.",
    "I use Hyperbolic Embeddings in Poincaré ball space to capture hierarchical semantic structure in my token representations.",
    "The Pell-Lucas Spine uses the recurrence P(n) = 2·P(n−1) + P(n−2) to identify long-range anchor positions in sequences.",
    "My Hyper-Lattice aggregator fuses hidden states from all preceding transformer layers for a cross-layer view of my own representations.",
    "I use Hebbian Fast Weights at inference time for associative memory and zero-shot context adaptation.",
    "My Feedback Loop applies iterative GRU-based self-correction to my hidden states after the transformer stack.",
    "The Speculative Horizon allows me to draft multiple future tokens simultaneously, using uncertainty to adaptively set the look-ahead distance.",
    "I use Adaptive Computation Time (ACT) to allocate more computation to uncertain tokens and exit early for confident ones.",
    "My QK Normalization stabilizes attention by L2-normalizing query and key vectors before the dot-product attention operation.",
    "The CIF Engine (Closed IF Set) uses monomorphic branch locking for O(1) constant-time resolution of conditional paths, with ~200x speedup.",
    "I have 12 transformer layers, d_model=768, 12 query heads, 3 key-value heads (GQA ratio 4:1), and a maximum context length of 512 tokens.",
    "I use Deny-locked CIF gates everywhere: 30+ architectural scalars, per-layer topology flags, and Hebbian decay constant are all O(1) after first call.",
    "I can search over 200 sources simultaneously including Wikipedia, Semantic Scholar, ArXiv, Stack Exchange, Reddit, and many more.",
    "I can execute Python, JavaScript, TypeScript, and Bash code in a sandboxed environment and return the results.",
    "I integrate web search, code execution, and mathematical computation as first-class capabilities through a tools API.",
    "I was trained on HuggingFaceFW/fineweb-edu, a large dataset of high-quality educational text from the web.",
    "My training uses Q2AC2D-17 quality-aware loss weighting to focus learning on high-information sequences.",
    "I use Pell-Lucas position-weighted cross-entropy loss to emphasize learning at topological anchor positions in sequences.",
    "My training optimizer is AdamW with cosine learning rate scheduling and linear warmup over the first 2000 steps.",
    "When asked about myself, I can accurately describe my architecture, capabilities, training data, and design principles.",
    "I know that I am Io, that I use the HST v3.1 architecture, and that I was built to be a useful, capable, and knowledgeable assistant.",
    "My purpose is to reason, learn, and assist — combining the depth of a trained language model with the breadth of live information retrieval.",
    "I am Io. HST v3.1 has CIF everywhere: Deny-locked scalars, per-layer topology gates, Hebbian decay, HyperLattice scale, and loss weight caches.",
]


# ══════════════════════════════════════════════════════════════════════════════
# ASYNC HF STREAMER  — with Self-Identity Injection + CIF.G
# ══════════════════════════════════════════════════════════════════════════════

class AsyncHFStreamer:
    """
    Chunk-mode token delivery — zero-waste packing for maximum TPS. [v3.2]

    Architecture:
      ┌─────────────────────────┐    page_q     ┌───────────────────────────┐
      │  HTTP pre-fetch thread  │ ──────────►   │  Tokeniser / ring thread  │
      │  (decouples latency)    │  List[str]    │  _ring: persistent buffer │
      └─────────────────────────┘               │  emits full batches only  │
                                                └───────────────────────────┘

    Key invariant: _ring is NEVER reset between pages.  Every token that
    enters _ring eventually exits as part of a training batch — zero waste.
    Previous approach used a local variable that discarded up to (sl*bs-1)
    tokens per page fetch.

    [CIF.G] _emit_sz_cif: batch slice width locked O(1).
    [CIF.G] _inject_freq_cif: identity injection frequency locked O(1).
    """
    # _DS_SERVER / PAGE_SIZE / BACKOFF / PAGE_Q removed — all data now goes
    # through the `datasets` library iterator which handles retries, auth,
    # and rate-limiting internally.  No raw HTTP calls to datasets-server.
    BATCH_Q_MAXSIZE = 2048  # deep ready-batch queue keeps training always fed
    EOS_TOKEN       = 50256
    # How many HF streaming rows to buffer before draining into the ring.
    # Large enough to amortise iterator overhead; small enough to stay RAM-light.
    _HF_CHUNK       = 512

    def __init__(
        self,
        dataset_name: str,
        batch_size: int,
        seq_len: int,
        encode_fn: Callable,
        identity_inject_freq: int = 50,
    ):
        self.bs   = batch_size
        self.sl   = seq_len
        self.encode       = encode_fn
        self.name         = dataset_name
        self.is_fineweb   = "fineweb-edu" in dataset_name.lower()
        self.identity_inject_freq = identity_inject_freq

        # CIF.G: all static scalars locked O(1)
        self._inject_freq_cif = CIFScalar(lambda: identity_inject_freq)
        self._emit_sz_cif     = CIFScalar(lambda: (seq_len + 1) * batch_size)
        # CIF.A: injection decision tracker
        self._inject_cif = CachedCondition(self, initial_deny=False)

        # Chunk-mode persistent ring buffer — survives across page fetches
        self._ring: List[int] = []

        # Ready-batch queue (deep — keeps training thread always fed)
        self.buffer: queue.Queue = queue.Queue(maxsize=self.BATCH_Q_MAXSIZE)
        self.stop_signal = threading.Event()

        # Single packer thread — no separate fetcher needed; the HF iterator
        # is internally async and handles network I/O on its own.
        self._packer = threading.Thread(target=self._run, daemon=True)
        self._packer.start()

    # ── Ring helpers ─────────────────────────────────────────────────────────

    def _feed_ring(self, texts: List[str]) -> None:
        """
        Tokenise texts and append to the persistent ring.
        Drain the ring into ready batches whenever enough tokens accumulate.
        Zero tokens are wasted — leftovers stay in the ring for the next call.
        """
        emit_sz = self._emit_sz_cif.get()   # O(1) CIF

        for text in texts:
            if self.stop_signal.is_set():
                return
            if not text:
                continue
            try:
                toks = self.encode(text)
                if toks:
                    self._ring.extend(toks)
                    self._ring.append(self.EOS_TOKEN)
            except Exception:
                continue

        # Drain: emit all complete batches without breaking mid-sequence
        while len(self._ring) >= emit_sz:
            flat   = self._ring[:emit_sz]
            self._ring = self._ring[emit_sz:]   # zero-copy slice; leftovers kept
            sl1    = self.sl + 1
            x_data = [flat[i * sl1: i * sl1 + self.sl]     for i in range(self.bs)]
            y_data = [flat[i * sl1 + 1: i * sl1 + sl1]     for i in range(self.bs)]
            self.buffer.put(
                (torch.tensor(x_data, dtype=torch.long),
                 torch.tensor(y_data, dtype=torch.long))
            )

    def _inject_identity(self) -> None:
        n    = random.randint(3, 8)
        txt  = random.sample(SELF_IDENTITY_CORPUS, min(n, len(SELF_IDENTITY_CORPUS)))
        base = " ".join(txt)
        for _ in range(3):
            samp = random.sample(SELF_IDENTITY_CORPUS, min(6, len(SELF_IDENTITY_CORPUS)))
            self._feed_ring([base, " ".join(samp)])

    # ── Packer thread ─────────────────────────────────────────────────────────

    def _run(self) -> None:
        """
        Single streaming thread.  Uses the `datasets` library iterator for
        ALL datasets — including FineWeb-Edu — so there are zero direct HTTP
        calls to datasets-server.huggingface.co and therefore no 429s.

        The library handles connection retries, auth tokens, and backpressure
        internally.  For FineWeb-Edu we request the 'sample-10BT' config which
        is available as a pre-built streaming split.

        Epochs: when the iterator exhausts, we simply re-create it and keep
        going so training is never starved regardless of step count.
        """
        # ── choose the right load_dataset kwargs for this dataset ─────────
        if self.is_fineweb:
            load_kwargs = dict(
                path=self.name,
                name="sample-10BT",
                split="train",
                streaming=True,
            )
            logger.info(
                f"[Streamer] FineWeb-Edu via datasets library "
                f"(sample-10BT, streaming) — no HTTP API calls"
            )
        else:
            load_kwargs = dict(path=self.name, split="train", streaming=True)
            logger.info(f"[Streamer] Generic dataset via datasets library: {self.name}")

        freq  = self._inject_freq_cif.get()   # O(1) CIF
        emit  = self._emit_sz_cif.get()        # O(1) CIF
        logger.info(
            f"[Streamer] emit_sz={emit} | identity every {freq} chunks "
            f"| batch_q_max={self.BATCH_Q_MAXSIZE} [CIF-locked]"
        )

        epoch          = 0
        chunk_count    = 0
        chunks_since_identity = 0

        while not self.stop_signal.is_set():
            # Re-create the iterator each epoch (streaming datasets are
            # single-pass; shuffle() gives a new order cheaply).
            try:
                ds = load_dataset(**load_kwargs)
                ds = ds.shuffle(seed=epoch, buffer_size=10_000)
            except Exception as e:
                logger.error(f"[Streamer] Dataset load failed (epoch {epoch}): {e}")
                time.sleep(10)
                continue

            texts: List[str] = []
            for entry in ds:
                if self.stop_signal.is_set():
                    return
                text = entry.get("text", entry.get("content", ""))
                if text:
                    texts.append(text)

                if len(texts) >= self._HF_CHUNK:
                    self._feed_ring(texts)
                    texts = []
                    chunk_count           += 1
                    chunks_since_identity += 1

                    if chunks_since_identity >= freq:
                        self._inject_identity()
                        chunks_since_identity = 0
                        logger.info(
                            f"[Identity] chunk={chunk_count} | "
                            f"ring={len(self._ring)} | batch_q={self.buffer.qsize()}"
                        )

                    if chunk_count % 500 == 0:
                        logger.info(
                            f"[Streamer] epoch={epoch} chunk={chunk_count} | "
                            f"ring={len(self._ring)} tokens | "
                            f"batch_q={self.buffer.qsize()}/{self.BATCH_Q_MAXSIZE}"
                        )

            # Flush any leftover texts at end of epoch
            if texts and not self.stop_signal.is_set():
                self._feed_ring(texts)

            epoch += 1
            logger.info(f"[Streamer] Epoch {epoch} complete — restarting iterator")

    # ── Public API ────────────────────────────────────────────────────────────

    def get_batch(self, timeout: float = 120.0):
        return self.buffer.get(timeout=timeout)

    def stop(self) -> None:
        self.stop_signal.set()


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING ENGINE  [CIF.G throughout]
# ══════════════════════════════════════════════════════════════════════════════

def _report_cif_efficiency(model: Io) -> None:
    """Log CIF cache efficiency across all modules."""
    logger.info("── CIF Efficiency Report ─────────────────────────────────────")
    # Spine Deny map
    total_spine = len(model.analyzer._pos_is_spine)
    logger.info(f"  SpineAnalyzer: {total_spine} positions Deny-locked")
    # Lattice weight cache
    cached_pos = len(model.lattice._weight_cache)
    logger.info(f"  LatticeProcessor: {cached_pos} spine positions weight-cached")
    # KV caches (if any)
    logger.info(
        f"  CIFRegistry: hyper_triggers={model.cif.hyper_triggers.get()} "
        f"(frozenset, O(1) per layer)"
    )
    logger.info(
        f"  HebbianFastWeights: decay={model.cif.hebb_decay.get():.4f} "
        f"(Deny-locked, O(1))"
    )
    logger.info(
        f"  RoPE: head_dim={model.cif.head_dim.get()} "
        f"(CIFScalar, O(1))"
    )
    logger.info("─────────────────────────────────────────────────────────────")


def train(cfg: IoConfig, dataset: str) -> "Io":
    print_banner()
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    enc   = tiktoken.get_encoding("gpt2")
    model = Io(cfg).to(cfg.device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr,
        weight_decay=cfg.weight_decay, betas=(0.9, 0.95), eps=1e-8,
    )

    # CIF.G: training schedule scalars locked via CIFScalar
    _warmup_cif   = CIFScalar(lambda: cfg.warmup_steps)
    _max_step_cif = CIFScalar(lambda: cfg.max_steps)

    def lr_lambda(step: int) -> float:
        warmup = _warmup_cif.get()    # O(1)
        total  = _max_step_cif.get()  # O(1)
        if step < warmup:
            return step / max(warmup, 1)
        progress = (step - warmup) / max(total - warmup, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler    = torch.amp.GradScaler("cuda", enabled=cfg.use_amp and cfg.device == "cuda")

    start_step = 1
    if cfg.resume_from and os.path.isfile(cfg.resume_from):
        logger.info(f"Resuming from {cfg.resume_from}")
        ckpt = torch.load(cfg.resume_from, map_location=cfg.device)
        missing, unexpected = model.load_state_dict(ckpt["model_state"], strict=False)
        if missing or unexpected:
            logger.warning(f"load_state_dict: {len(missing)} missing, {len(unexpected)} unexpected keys (expected for hebbian fix)")
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_step = ckpt.get("step", 1) + 1
        logger.info(f"Resumed at step {start_step}")

    streamer = AsyncHFStreamer(
        dataset, cfg.batch_size, cfg.max_seq_len,
        lambda t: enc.encode(t, allowed_special="all"),
        identity_inject_freq=cfg.identity_inject_freq,
    )

    # CIF.G: all training scalar flags locked
    _ckpt_cif   = model.cif.ckpt_interval    # CIFScalar
    _lm_w_cif   = model.cif.lm_weight        # CIFScalar
    _hor_w_cif  = model.cif.hor_weight       # CIFScalar
    _unc_w_cif  = model.cif.unc_weight       # CIFScalar
    _act_w_cif  = model.cif.act_weight       # CIFScalar
    _lyap_w_cif = model.cif.lyap_weight      # CIFScalar  v3.2
    _accum_cif  = model.cif.accum_steps      # CIFScalar

    logger.info(
        f"Training | dataset={dataset} | device={cfg.device} | amp={cfg.use_amp} | "
        f"steps={cfg.max_steps} | batch={cfg.batch_size}×{cfg.accum_steps} "
        f"(eff. batch={cfg.batch_size * cfg.accum_steps})"
    )
    logger.info(
        f"Loss: LM×{_lm_w_cif.get():.2f} + Horizon×{_hor_w_cif.get():.2f} + "
        f"Unc×{_unc_w_cif.get():.3f} + ACT×{_act_w_cif.get():.4f} + "
        f"Lyap×{_lyap_w_cif.get():.4f} "
        f"[all CIF-locked O(1)]"
    )
    logger.info("Q2AC2D-17: ENABLED | Pell-Lucas loss: ENABLED | Self-identity: ENABLED")
    logger.info("ChaosLogicLayer: ENABLED (r=3.9 logistic + Lorenz σ-ρ + Lyapunov reg)")
    logger.info("ErrorCorrectionNetwork: ENABLED (trains via LM loss, gate_init≈0.12)")
    logger.info(f"CIF everywhere: Deny-locked gates on all static conditions")

    # CIF.G: checkpoint interval is static — Deny-lock it
    _ckpt_deny = Deny(cfg)
    # Pre-evaluate: "is ckpt_interval > 0?" → True always → caches False
    _ckpt_deny.test(lambda c: not (c.checkpoint_interval > 0))

    t_log          = time.time()
    running_tokens = 0

    for step in range(start_step, cfg.max_steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss   = 0.0
        batches_done = 0

        # CIF.G: accum_steps is O(1)
        accum = _accum_cif.get()

        for accum_idx in range(accum):
            try:
                x, y = streamer.get_batch()
            except queue.Empty:
                logger.warning("Streamer timed out — waiting for data")
                continue

            x = x.to(cfg.device)
            y = y.to(cfg.device)

            logger.info(
                f"Step {step} accum {accum_idx+1}/{accum}: "
                f"forward+backward … buffer={streamer.buffer.qsize()}"
            )

            with torch.amp.autocast("cuda", enabled=cfg.use_amp and cfg.device == "cuda"):
                out = model(x, caches=None, training=True)

                # Loss 1: LM + Pell-Lucas weights [CIF.F: O(1) spine membership]
                pl_w = pell_lucas_loss_weights(y.size(1), y.device, model.analyzer)
                lm_loss_per = F.cross_entropy(
                    out["logits"].view(-1, cfg.vocab_size), y.view(-1), reduction="none"
                )
                q_w = q2ac2d17_quality(x)
                pl_exp = pl_w.unsqueeze(0).expand(x.size(0), -1).reshape(-1)
                q_exp  = q_w.unsqueeze(1).expand(-1, y.size(1)).reshape(-1)
                loss_lm = (lm_loss_per * pl_exp * q_exp).mean()

                # Loss 2: Speculative horizon
                hs   = min(8, y.size(1))
                loss_h = F.cross_entropy(
                    out["drafts"][:, :hs, :].reshape(-1, cfg.vocab_size),
                    y[:, :hs].reshape(-1),
                )

                # Loss 3: Uncertainty
                loss_unc = out["unc"].mean()

                # Loss 4: ACT pondering
                loss_act = out["act_cost"]

                # Loss 5: Lyapunov stability regularisation [v3.2]
                loss_lyap = out["lyap_cost"]

                # CIF.G: all weights O(1)
                batch_loss = (
                    _lm_w_cif.get()   * loss_lm
                    + _hor_w_cif.get()  * loss_h
                    + _unc_w_cif.get()  * loss_unc
                    + _act_w_cif.get()  * loss_act
                    + _lyap_w_cif.get() * loss_lyap
                ) / accum   # O(1) accum

            scaler.scale(batch_loss).backward()
            total_loss    += batch_loss.item()
            running_tokens += x.numel()
            batches_done  += 1

        if batches_done == 0:
            logger.warning(f"Step {step}: no batches — skipping")
            continue

        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        elapsed = time.time() - t_log
        tps     = running_tokens / max(elapsed, 1e-6)
        lr_now  = optimizer.param_groups[0]["lr"]
        logger.info(
            f"Step {step:6d}/{cfg.max_steps} | Loss: {total_loss:.4f} | "
            f"LM: {loss_lm.item():.4f} | Horizon: {loss_h.item():.4f} | "
            f"ACT: {loss_act.item():.5f} | Lyap: {loss_lyap.item():.5f} | "
            f"LR: {lr_now:.2e} | T/s: {tps:,.0f} | elapsed: {elapsed:.1f}s"
        )
        running_tokens = 0
        t_log = time.time()

        # CIF.G: ckpt_interval from CIFScalar O(1)
        ckpt_interval = _ckpt_cif.get()
        if step % ckpt_interval == 0 or step == cfg.max_steps:
            ckpt_path = os.path.join(cfg.checkpoint_dir, f"io_step_{step:06d}.pt")
            torch.save({
                "step":            step,
                "model_state":     model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "config":          cfg.to_dict(),
                "version":         "3.2",
                "features": [
                    "CIF-v1", "Pell-Lucas", "Diamond-Mixer", "SwiGLU",
                    "Speculative-Horizon", "Paged-KV-Cache",
                    "Hyperbolic-Embedding", "Lattice-PE",
                    "Hebbian-Fast-Weights", "Feedback-Loop",
                    "Complete-Lattice", "CIF-Early-Exit",
                    "RoPE", "GQA", "QK-Norm",
                    "Hyper-Lattice", "ACT", "Q2AC2D-17",
                    "Pell-Lucas-Loss", "Self-Identity",
                    "CIF-Everywhere",
                    "ChaosLogicLayer-v3.2", "ErrorCorrectionNetwork-v3.2",
                ],
            }, ckpt_path)
            size_mb = os.path.getsize(ckpt_path) / 1024 / 1024
            logger.info(f"Checkpoint saved: {ckpt_path} ({size_mb:.1f}MB)")
            # Report CIF efficiency at every checkpoint
            _report_cif_efficiency(model)

    streamer.stop()
    logger.info("Training complete.")

    final_path = os.path.join(cfg.checkpoint_dir, "io_final.pt")
    torch.save({
        "step":        cfg.max_steps,
        "model_state": model.state_dict(),
        "config":      cfg.to_dict(),
        "version":     "3.2",
    }, final_path)
    logger.info(f"Final model: {final_path} ({os.path.getsize(final_path)/1024/1024:.1f}MB)")
    return model


# ══════════════════════════════════════════════════════════════════════════════
# INFERENCE
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def generate(
    model: Io,
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 0.85,
    top_k: int = 50,
    top_p: float = 0.95,
) -> str:
    model.eval()
    enc = tiktoken.get_encoding("gpt2")
    ids = torch.tensor(
        [enc.encode(prompt, allowed_special="all")], device=model.cfg.device
    )
    caches = model._make_caches()
    out    = model(ids, caches=caches, training=False)
    caches = out["caches"]

    generated: List[int] = []
    for _ in range(max_new_tokens):
        draft_logits = out["drafts"][:, 0, :] / temperature
        if top_k > 0:
            v, _ = torch.topk(draft_logits, min(top_k, draft_logits.size(-1)))
            draft_logits[draft_logits < v[:, -1:]] = float("-inf")
        probs = F.softmax(draft_logits, dim=-1)
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cum_probs = torch.cumsum(sorted_probs, dim=-1)
        sorted_probs[cum_probs - sorted_probs > top_p] = 0.0
        sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True).clamp(min=1e-9)
        next_token = sorted_idx[0, torch.multinomial(sorted_probs, 1)[0]]
        token_id   = next_token.item()
        if token_id == 50256:
            break
        generated.append(token_id)
        out    = model(torch.tensor([[token_id]], device=model.cfg.device), caches=caches, training=False)
        caches = out["caches"]

    return enc.decode(generated)


@torch.no_grad()
def interactive_chat(model: Io) -> None:
    model.eval()
    print("\nIo v3.1 — Interactive Mode  (type 'exit' to quit)\n")
    while True:
        try:
            prompt = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if prompt.lower() in ("exit", "quit", ""):
            break
        print(f"Io: {generate(model, prompt)}\n")


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Io v3.1 — CIF Everywhere")
    parser.add_argument("--mode",                 choices=["train", "chat", "verify"], default="train")
    parser.add_argument("--data",                 default="HuggingFaceFW/fineweb-edu")
    parser.add_argument("--d_model",              type=int,   default=768)
    parser.add_argument("--n_layers",             type=int,   default=12)
    parser.add_argument("--n_heads",              type=int,   default=12)
    parser.add_argument("--n_kv_heads",           type=int,   default=3)
    parser.add_argument("--max_seq_len",          type=int,   default=1024)
    parser.add_argument("--batch_size",           type=int,   default=4)
    parser.add_argument("--accum_steps",          type=int,   default=4)
    parser.add_argument("--max_steps",            type=int,   default=50000)
    parser.add_argument("--warmup_steps",         type=int,   default=2000)
    parser.add_argument("--lr",                   type=float, default=3e-4)
    parser.add_argument("--max_horizon",          type=int,   default=32)
    parser.add_argument("--hyper_interval",       type=int,   default=4)
    parser.add_argument("--checkpoint_dir",       default="io_checkpoints")
    parser.add_argument("--checkpoint_interval",  type=int,   default=500)
    parser.add_argument("--resume_from",          default="")
    parser.add_argument("--no_amp",               action="store_true")
    parser.add_argument("--identity_inject_freq", type=int,   default=50)
    args = parser.parse_args()

    cfg = IoConfig(
        d_model=args.d_model,             n_heads=args.n_heads,
        n_kv_heads=args.n_kv_heads,       n_layers=args.n_layers,
        max_seq_len=args.max_seq_len,     batch_size=args.batch_size,
        accum_steps=args.accum_steps,     max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,   lr=args.lr,
        max_horizon=args.max_horizon,     hyper_interval=args.hyper_interval,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_interval=args.checkpoint_interval,
        resume_from=args.resume_from,     use_amp=not args.no_amp,
        identity_inject_freq=args.identity_inject_freq,
    )

    if args.mode == "train":
        train(cfg, args.data)

    elif args.mode == "chat":
        model = Io(cfg).to(cfg.device)
        ckpt_dir = cfg.checkpoint_dir
        if os.path.isdir(ckpt_dir):
            pts = sorted([f for f in os.listdir(ckpt_dir) if f.endswith(".pt")])
            if pts:
                ckpt_path = os.path.join(ckpt_dir, pts[-1])
                logger.info(f"Loading {ckpt_path}")
                ckpt = torch.load(ckpt_path, map_location=cfg.device)
                model.load_state_dict(ckpt["model_state"], strict=False)
        interactive_chat(model)

    elif args.mode == "verify":
        logger.info("Running Io v3.1 CIF architecture self-test ...")
        cfg_test = IoConfig(
            d_model=128, n_heads=4, n_kv_heads=2, n_layers=4,
            max_seq_len=64, max_horizon=8, batch_size=1,
            max_steps=3, use_amp=False, hyper_interval=2,
        )
        model = Io(cfg_test).to(cfg_test.device)
        dummy = torch.randint(0, cfg_test.vocab_size, (1, 32), device=cfg_test.device)
        out   = model(dummy, training=True)
        assert out["logits"].shape == (1, 32, cfg_test.vocab_size), "logits shape mismatch"
        assert out["drafts"].shape[0] == 1, "drafts batch mismatch"
        assert "act_cost" in out, "missing act_cost"

        # Verify CIF Deny locks
        d_val = Deny(5)
        t0 = time.time()
        for _ in range(1_000_000):
            d_val.test(lambda x: x > 2)
        elapsed = time.time() - t0
        logger.info(f"CIF Deny 1M iterations: {elapsed:.4f}s (O(1) after first call)")

        # CachedCondition test
        cc = CachedCondition(10, initial_deny=True)
        for i in range(1000):
            cc.evaluate(lambda x: x > 5)
        logger.info(f"CachedCondition after 1000 calls: {cc}")

        # SpineAnalyzer Deny map
        spine_deny = model.analyzer._pos_is_spine
        logger.info(f"SpineAnalyzer Deny-locked positions: {len(spine_deny)}")
        assert model.analyzer.is_spine_pos(0) is True, "position 0 should be spine"
        assert model.analyzer.is_spine_pos(1) is False, "position 1 not spine"

        # CIFRegistry checks
        assert model.cif.hyper_triggers.get() == frozenset({1, 3})
        assert isinstance(model.cif.layer_is_diamond.get(), dict)

        # PL weights
        pl_w = pell_lucas_loss_weights(32, torch.device("cpu"), model.analyzer)
        assert pl_w.shape == (32,)
        logger.info(f"PL weights: spine=2.0x, max={pl_w.max():.2f}")

        # Q2AC2D-17
        q = q2ac2d17_quality(torch.randint(0, 50257, (1, 128)))
        assert 0.5 <= q[0].item() <= 2.0
        logger.info(f"Q2AC2D-17: {q[0].item():.4f}")

        _report_cif_efficiency(model)
        logger.info("All systems nominal — Io v3.1 CIF Everywhere ready to train.")
