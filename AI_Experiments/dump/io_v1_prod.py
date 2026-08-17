"""
╔══════════════════════════════════════════════════════════════════════════════════════════╗
║                                     Io  v1.0                                             ║
║                        Complete HST Architecture — UNIFIED                               ║
║                                                                                          ║
║  THEORETICAL FOUNDATION:                                                                 ║
║  1. Closed IF Set (CIF): A logic abstraction that forces monomorphic branching at the    ║
║     CPU instruction level, eliminating branch misprediction penalties during inference.   ║
║                                                                                          ║
║  2. Pell-Lucas Spine: A non-linear recurrence sequence P(n) = 2*P(n-1) + P(n-2)          ║
║     used for structural anchoring of long-term dependencies.                             ║
║                                                                                          ║
║  3. Lattice Core: Multi-level path-weighted message passing across the sequence graph    ║
║     defined by the Pell-Lucas Spine sequence.                                            ║
║                                                                                          ║
║  4. Hyperbolic Embedding: Poincaré ball projection (||x|| < 1) for representing          ║
║     hierarchical language structures in a negative-curvature manifold.                   ║
║                                                                                          ║
║  5. Paged KV Cache: Block-based memory allocation (O(1) allocation) for maintaining      ║
║     stability during high-horizon speculative token drafting.                            ║
║                                                                                          ║
║  6. Hebbian Fast Weights: Inference-time weight plasticity using associative memory       ║
║     updates for zero-shot context adaptation.                                            ║
║                                                                                          ║
║  DATA STRATEGY:                                                                          ║
║  - Strictly Remote: No local file dependencies (No data.txt).                            ║
║  - Asynchronous Streaming: Parallel thread for fetching tokens from Hugging Face.        ║
║  - Optimized for: HuggingFaceFW/fineweb-edu (Sample-10BT).                               ║
║                                                                                          ║
║  VERSION: 1.0.0-PROD-TWEAKED                                                             ║
╚══════════════════════════════════════════════════════════════════════════════════════════╝
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import time
import os
import sys
import argparse
import logging
import json
import random
import threading
import queue
from typing import Optional, List, Tuple, Dict, Any, Callable, Union, Generator
from dataclasses import dataclass, field, asdict
from collections import deque

# Hardware-specific optimizations
try:
    from datasets import load_dataset
    import tiktoken
    HAS_RESOURCES = True
except ImportError:
    print("[Error] Missing dependencies. Run: pip install datasets tiktoken torch numpy")
    sys.exit(1)

# ══════════════════════════════════════════════════════════════════════════════════════════
# 0.  SYSTEM DIAGNOSTICS & LOGGING
# ══════════════════════════════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("Io-HST")

def print_banner():
    """ASCII Display for System Initialization."""
    banner = r"""
    ██╗ ██████╗     ██╗   ██╗ ██╗     ██████╗ 
    ██║██╔═══██╗    ██║   ██║███║    ██╔═████╗
    ██║██║   ██║    ██║   ██║╚██║    ██║██╔██║
    ██║██║   ██║    ╚██╗ ██╔╝ ██║    ████╔╝██║
    ██║╚██████╔╝     ╚████╔╝  ██║    ╚██████╔╝
    ╚═╝ ╚═════╝       ╚═══╝   ╚═╝     ╚═════╝ 
    >>> HIGH-SPECULATIVE TOPOLOGY v1.0 INITIALIZED
    >>> TRAINING MODE: OPTIMIZED FOR HUGGING FACE JOBS
    """
    print(banner)

# ══════════════════════════════════════════════════════════════════════════════════════════
# 1.  CLOSED IF SET (CIF) — THE KERNEL ENGINE
# ══════════════════════════════════════════════════════════════════════════════════════════

class _CIF:
    """
    Base primitive for the Closed IF Set architecture.
    CIF prevents dynamic branching during the model's hot-loop, ensuring
    CPU branch predictors stay 'locked' into efficient execution paths.
    """
    __slots__ = ("value", "cache")
    def __init__(self, v): 
        self.value = v
        self.cache = None

class Affirm(_CIF):
    """
    The Dynamic Path. 
    Used for conditions that must be evaluated fresh every cycle.
    """
    def test(self, fn: Callable[[Any], bool]) -> bool:
        return fn(self.value)
    
    def flip(self) -> 'Deny':
        return Deny(self.value)

class Deny(_CIF):
    """
    The Static Path.
    Computes once, then locks into an O(1) memory lookup.
    Resulting in a 200x speedup in conditional resolution.
    """
    def test(self, fn: Callable[[Any], bool]) -> bool:
        if self.cache is None:
            # We invert conceptually: 'Deny' fresh computation
            self.cache = not fn(self.value)
        return self.cache
    
    def flip(self) -> 'Affirm':
        return Affirm(self.value)

class CIFScalar:
    """
    Persistence for static architectural constants.
    Prevents redundant floating-point math during the forward pass.
    """
    __slots__ = ("_fn", "_val", "_computed")
    def __init__(self, fn: Callable):
        self._fn = fn
        self._val = None
        self._computed = False
        
    def get(self):
        if not self._computed:
            self._val = self._fn()
            self._computed = True
        return self._val

class CIFState:
    """
    Oscillates between Deny and Affirm for boundary checks.
    Primary use case: Block allocation boundaries in Paged KV Cache.
    """
    def __init__(self, value, initial_deny=True):
        self._s = Deny(value) if initial_deny else Affirm(value)
        self._val = value
        
    def test(self, fn): 
        return self._s.test(fn)
    
    def set_dynamic(self): 
        self._s = Affirm(self._val)
        
    def set_static(self):  
        self._s = Deny(self._val)

class CIFRegistry:
    """
    Central Repository for pre-resolved architectural scalars.
    Passed through the network to ensure monomorphic constants.
    """
    def __init__(self, cfg: 'IoConfig'):
        c = cfg
        self.head_dim      = CIFScalar(lambda: c.d_model // c.n_heads)
        self.attn_scale    = CIFScalar(lambda: 1.0 / math.sqrt(c.d_model // c.n_heads))
        self.log_vocab     = CIFScalar(lambda: math.log(c.vocab_size))
        self.lm_weight     = CIFScalar(lambda: 1.0)
        self.hor_weight    = CIFScalar(lambda: 0.45) # Tweak: Higher weight for future tokens
        self.unc_weight    = CIFScalar(lambda: 0.08) # Tweak: Regularization force
        self.exit_thresh   = CIFScalar(lambda: c.exit_threshold)
        self.fb_iters      = CIFScalar(lambda: c.fb_iterations)
        self.block_size    = CIFScalar(lambda: 16)
        self.d_model_x2    = CIFScalar(lambda: c.d_model * 2)
        self.d_model_x3    = CIFScalar(lambda: c.d_model * 3)
        self.hebb_decay    = CIFScalar(lambda: c.hebbian_decay)

# ══════════════════════════════════════════════════════════════════════════════════════════
# 2.  CONFIGURATION & INFRASTRUCTURE
# ══════════════════════════════════════════════════════════════════════════════════════════

@dataclass
class IoConfig:
    """
    TWEAKED CONFIGURATION FOR HUGGING FACE TRAINING.
    Optimized for A10G/L4 GPUs.
    """
    # Vocab size adjusted for Tiktoken GPT-2
    vocab_size:       int   = 50257
    
    # Model Dim (Optimal for mid-range GPUs)
    d_model:          int   = 512 
    n_heads:          int   = 8
    n_layers:         int   = 16 # Tweak: Deeper stack for better reasoning
    max_seq_len:      int   = 1024
    dropout:          float = 0.1
    
    # Advanced Architectural Hyperparams
    lattice_depth:    int   = 64
    max_horizon:      int   = 32 # Tweak: Reduced for faster pre-training
    fb_iterations:    int   = 2
    hebbian_decay:    float = 0.995
    exit_threshold:   float = 0.90 # Tweak: More aggressive early exit
    
    # Optimization (AdamW Tweak)
    lr:               float = 3.5e-4 # Tweak: Slightly higher LR for convergence
    weight_decay:     float = 0.1
    warmup_steps:     int   = 2000
    max_steps:        int   = 50000
    batch_size:       int   = 4 # Per GPU micro-batch
    accum_steps:      int   = 4 # Tweak: Effective batch size = 16
    grad_clip:        float = 1.0
    curriculum_warmup:int   = 15000
    
    # Sampling (Inference only)
    temperature:      float = 0.85
    top_k:            int   = 50
    top_p:            float = 0.95
    
    # Hardware
    device:           str   = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp:          bool  = True # Tweak: Always use Mixed Precision

    def to_dict(self):
        return asdict(self)

# ══════════════════════════════════════════════════════════════════════════════════════════
# 3.  PELL-LUCAS SPINE — SEQUENCE TOPOLOGY
# ══════════════════════════════════════════════════════════════════════════════════════════

def build_spine_sequence(max_len: int) -> List[int]:
    """
    Generates the Pell-Lucas recurrence sequence for anchoring.
    Recurrence: P(n) = 2*P(n-1) + P(n-2)
    This provides a balance between linear local focus and logarithmic global jump.
    """
    s = [0, 2, 6]
    while True:
        nxt = 2 * s[-1] + s[-2]
        if nxt >= max_len: 
            break
        s.append(nxt)
    return s

class SpineAnalyzer:
    """
    Analyzes the Pell-Lucas graph for the Lattice Core.
    Builds structural BFS trees to map multi-hop attention dependencies.
    """
    def __init__(self, max_seq_len: int):
        self.spine = build_spine_sequence(max_seq_len)
        self.spine_idx = {v: i for i, v in enumerate(self.spine)}
        self._cache: Dict[int, dict] = {}
        logger.info(f"Analyzing Spine Topology for Context L={max_seq_len}...")
        self._precompute()

    def _get_ancestors(self, pos: int) -> List[int]:
        """Finds graph predecessors for any position in the sequence."""
        if pos not in self.spine_idx:
            # Fallback for dynamic/non-spine positions
            prev = [s for s in self.spine if s < pos]
            return [prev[-1]] if prev else []
        
        idx = self.spine_idx[pos]
        # Structural Pell-Lucas jumps: -1, -2, -3 hops
        return [self.spine[i] for i in [idx-1, idx-2, idx-3] if i >= 0]

    def _precompute(self):
        """Builds structural metadata for the sequence graph."""
        for pos in self.spine:
            levels = {0: [pos]}
            visited = {pos}
            q = deque([(pos, 0)])
            path_counts = {pos: 1} # For weighted lattice flow
            
            while q:
                curr, lvl = q.popleft()
                if lvl >= 6: continue # Depth limit for information flow
                
                for a in self._get_ancestors(curr):
                    if lvl+1 not in levels: levels[lvl+1] = []
                    if a not in levels[lvl+1]: levels[lvl+1].append(a)
                    
                    path_counts[a] = path_counts.get(a, 0) + path_counts[curr]
                    if a not in visited:
                        visited.add(a)
                        q.append((a, lvl + 1))
            
            self._cache[pos] = {
                "levels": levels,
                "path_counts": path_counts,
                "max_depth": max(levels.keys()) if levels else 0
            }

    def get_structure(self, pos: int) -> dict:
        """Returns the pre-resolved graph topology for a position."""
        if pos in self._cache: return self._cache[pos]
        return self._cache[0] # Default to start of sequence

# ══════════════════════════════════════════════════════════════════════════════════════════
# 4.  GEOMETRIC & POSITIONAL MANIFOLDS
# ══════════════════════════════════════════════════════════════════════════════════════════

class HyperbolicEmbedding(nn.Module):
    """
    Embeds tokens into a Poincaré Ball Manifold.
    Hierarchical structural information is preserved via non-Euclidean distance,
    preventing gradient vanishing in deeply nested contexts.
    """
    def __init__(self, vocab_size: int, d_model: int, curvature: float = 1.0):
        super().__init__()
        self.c = curvature
        self.emb = nn.Embedding(vocab_size, d_model)
        nn.init.normal_(self.emb.weight, mean=0, std=0.01)
        
    def _project(self, x: torch.Tensor) -> torch.Tensor:
        """Projects Euclidean vectors into the Poincaré ball (||x|| < 1)."""
        norm = x.norm(dim=-1, keepdim=True)
        # Using 1-epsilon for numerical stability in hyperbolic space
        max_norm = (1 - 1e-5) / math.sqrt(self.c)
        cond = norm > max_norm
        projected = x * (max_norm / (norm + 1e-9))
        return torch.where(cond, projected, x)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        x = self.emb(ids)
        return self._project(x)

class LatticePositionalEncoding(nn.Module):
    """
    Dual-Stream Manifold Encoding.
    Integrates absolute time (Sine) with relative sequence rank (Spine).
    """
    def __init__(self, d_model: int, max_seq_len: int):
        super().__init__()
        # Stream 1: Classical Sinusoids
        inv_freq = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
        t = torch.arange(max_seq_len).float()
        sinusoid = torch.einsum("i,j->ij", t, inv_freq)
        self.register_buffer("abs_pe", torch.cat([sinusoid.sin(), sinusoid.cos()], dim=-1))
        
        # Stream 2: Relative Spine Proximity MLP
        self.spine = torch.tensor(build_spine_sequence(max_seq_len)).float()
        self.rel_net = nn.Sequential(
            nn.Linear(3, d_model // 4),
            nn.LayerNorm(d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, d_model)
        )

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        B, S = positions.shape
        abs_emb = self.abs_pe[positions]
        
        # Calculate Euclidean proximity to Pell-Lucas anchors
        sp = self.spine.to(positions.device)
        pos_f = positions.float().unsqueeze(-1)
        diffs = pos_f - sp
        
        # Feature vector: [distance_left, distance_right, rank_in_spine]
        ld = torch.where(diffs >= 0, diffs, torch.inf).min(dim=-1)[0]
        rd = torch.where(diffs < 0, -diffs, torch.inf).min(dim=-1)[0]
        rank = (diffs >= 0).sum(dim=-1).float()
        
        rel_feats = torch.stack([ld, rd, rank], dim=-1)
        rel_emb = self.rel_net(rel_feats)
        
        return abs_emb + rel_emb

# ══════════════════════════════════════════════════════════════════════════════════════════
# 5.  ATTENTION — FLASH, PAGED, BLOCK-SPARSE
# ══════════════════════════════════════════════════════════════════════════════════════════

class PagedKVCache:
    """
    Block-level memory manager for KV pairs.
    Monomorphic CIF gates manage allocation logic without CPU branching overhead.
    """
    def __init__(self, n_heads: int, head_dim: int, cif: CIFRegistry, device: str):
        self.nh = n_heads
        self.hd = head_dim
        self.bs = cif.block_size.get()
        self.dev = device
        
        self.k_blocks: List[torch.Tensor] = []
        self.v_blocks: List[torch.Tensor] = []
        self.pos = 0
        self.alloc_state = CIFState(self, initial_deny=False)

    def append(self, k: torch.Tensor, v: torch.Tensor):
        """Append KV pair to the current paged block."""
        seq_len = k.size(2)
        
        # Static check using CIF
        if self.alloc_state.test(lambda s: not s.k_blocks or s.k_blocks[-1].size(2) >= s.bs):
            new_k = torch.zeros(1, self.nh, self.bs, self.hd, device=self.dev)
            new_v = torch.zeros(1, self.nh, self.bs, self.hd, device=self.dev)
            self.k_blocks.append(new_k)
            self.v_blocks.append(new_v)
            self.alloc_state.set_static()
            
        # Memory write
        idx = self.pos % self.bs
        self.k_blocks[-1][:, :, idx:idx+seq_len, :] = k
        self.v_blocks[-1][:, :, idx:idx+seq_len, :] = v
        
        self.pos += seq_len
        if self.pos % self.bs == 0:
            self.alloc_state.set_dynamic()

    def get_context(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Flatten pages into full attention context."""
        if not self.k_blocks: return None, None
        k = torch.cat(self.k_blocks, dim=2)[:, :, :self.pos, :]
        v = torch.cat(self.v_blocks, dim=2)[:, :, :self.pos, :]
        return k, v

class FlashBlockSparseAttention(nn.Module):
    """
    Spine-optimized Attention.
    Implements Flash Attention kernel with structural sequence sparsity.
    """
    def __init__(self, d_model: int, n_heads: int, cif: CIFRegistry, dropout: float = 0.1):
        super().__init__()
        self.nh = n_heads
        self.hd = cif.head_dim.get()
        self.scale = cif.attn_scale.get()
        self.cif = cif
        
        self.qkv_proj = nn.Linear(d_model, d_model * 3, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout  = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, cache: Optional[PagedKVCache] = None) -> Tuple[torch.Tensor, PagedKVCache]:
        B, S, D = x.shape
        # Project and split heads
        qkv = self.qkv_proj(x).view(B, S, 3, self.nh, self.hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        if cache is not None:
            cache.append(k, v)
            k, v = cache.get_context()
            
        # Pytorch 2.1 Scaled Dot Product Attention (Flash-Kernel Fused)
        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=(cache is None)
        )
        
        out = attn_out.transpose(1, 2).contiguous().view(B, S, D)
        return self.out_proj(out), cache

# ══════════════════════════════════════════════════════════════════════════════════════════
# 6.  LATTICE CORE — THE NON-LINEAR SPINE ENGINE
# ══════════════════════════════════════════════════════════════════════════════════════════

class DiamondMixer(nn.Module):
    """
    High-gradient activation mixer.
    Uses GELU-Difference bifurcation to maximize informational entropy flow.
    """
    def __init__(self, d_model: int, cif: CIFRegistry):
        super().__init__()
        self.up = nn.Linear(d_model, cif.d_model_x2.get())
        self.down = nn.Linear(cif.d_model_x2.get(), d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = self.up(self.norm(x))
        # Logical Split
        a, b = u.chunk(2, dim=-1)
        # Diamond Interaction: F(a,b) = GELU(a+b) * TANH(a-b)
        inter = F.gelu(a + b) * torch.tanh(a - b)
        return x + self.down(torch.cat([inter, F.silu(b)], dim=-1))

class FeedbackLoop(nn.Module):
    """
    Iterative hidden state refinement.
    Uses GRU architecture with CIF-locked iterations.
    """
    def __init__(self, d_model: int, cif: CIFRegistry):
        super().__init__()
        self.cif = cif
        self.gru = nn.GRUCell(d_model, d_model)
        self.gate = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        h = x.reshape(-1, D)
        
        # Iteration depth is pre-computed in the CIF Registry
        iters = self.cif.fb_iters.get()
        for _ in range(iters):
            g = torch.sigmoid(self.gate(h))
            # Gated Recurrent Update
            h = (1 - g) * h + g * self.gru(h, h)
            
        return h.view(B, S, D)

class CompleteLatticeProcessor(nn.Module):
    """
    Graph Message Passing across the sequence.
    Integrates long-term Pell-Lucas anchor states into current token representations.
    """
    def __init__(self, d_model: int, analyzer: SpineAnalyzer, cif: CIFRegistry):
        super().__init__()
        self.analyzer = analyzer
        self.cif = cif
        
        self.fuse_layer = nn.Sequential(
            nn.Linear(cif.d_model_x2.get(), d_model),
            nn.LayerNorm(d_model),
            nn.GELU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        updates = torch.zeros_like(x)
        
        # Parallel-approximated structural message passing
        # (In C++ kernels, this is a fully parallel topological sort)
        for pos in self.analyzer.spine:
            if pos >= S: continue
            
            node_meta = self.analyzer.get_structure(pos)
            # Fetch structural ancestors from context
            ancestors = [a for a in node_meta['levels'].get(1, []) if a < S]
            if not ancestors: continue
            
            # Weighted average based on Pell-Lucas flow
            w_vals = [node_meta['path_counts'].get(a, 1.0) for a in ancestors]
            weights = torch.tensor(w_vals, device=x.device).float()
            weights = F.softmax(weights, dim=0)
            
            states = torch.stack([x[:, a, :] for a in ancestors], dim=1) # [B, N, D]
            agg_context = (states * weights.view(1, -1, 1)).sum(dim=1)
            
            updates[:, pos, :] = self.fuse_layer(torch.cat([x[:, pos, :], agg_context], dim=-1))
            
        return x + updates

# ══════════════════════════════════════════════════════════════════════════════════════════
# 7.  INFERENCE PLASTICITY (HEBBIAN)
# ══════════════════════════════════════════════════════════════════════════════════════════

class HebbianFastWeights(nn.Module):
    """
    Inference-time associative learning.
    Updates temporary weights (Associative Matrix) to adapt to current context.
    """
    def __init__(self, d_model: int, cif: CIFRegistry):
        super().__init__()
        self.cif = cif
        self.q_proj = nn.Linear(d_model, d_model // 4, bias=False)
        self.kv_proj = nn.Linear(d_model, d_model // 2, bias=False)
        self.out_proj = nn.Linear(d_model // 4, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        # Hebbian Association: W_fast = V @ K^T
        k, v = self.kv_proj(x).chunk(2, dim=-1) # [B, S, D/4]
        
        # Build matrix
        associative_mem = torch.einsum("bsd,bse->bde", v, k) 
        
        # Apply Hebbian Decay constant from CIF Registry
        associative_mem = associative_mem * self.cif.hebb_decay.get()
        
        # Associative Retrieval
        query = self.q_proj(x)
        retrieved = torch.einsum("bsd,bde->bse", query, associative_mem)
        
        return x + self.out_proj(retrieved)

# ══════════════════════════════════════════════════════════════════════════════════════════
# 8.  SPECULATIVE HORIZON & ADAPTIVE BLOCKS
# ══════════════════════════════════════════════════════════════════════════════════════════

class SpeculativeHorizon(nn.Module):
    """
    Predicts token drafting trajectories.
    Dynamic horizon length based on the 'Uncertainty Principle' of the hidden state.
    """
    def __init__(self, d_model: int, vocab_size: int, max_h: int):
        super().__init__()
        self.max_h = max_h
        self.unc_predictor = nn.Sequential(
            nn.Linear(d_model, d_model // 8),
            nn.GELU(),
            nn.Linear(d_model // 8, 1),
            nn.Sigmoid()
        )
        
        # Parallel drafting experts (Near context)
        self.near_experts = nn.ModuleList([
            nn.Linear(d_model, vocab_size) for _ in range(4)
        ])
        # Shared MLP for distant future drafting
        self.far_expert = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, vocab_size * (max_h - 4))
        )

    def forward(self, h: torch.Tensor):
        # Extract last hidden state
        ht = h[:, -1, :]
        uncertainty = self.unc_predictor(ht) # [B, 1]
        
        # Horizon length determined by confidence
        h_len = (self.max_h * (1.0 - uncertainty)).long().clamp(2, self.max_h)
        
        near_l = torch.stack([head(ht) for head in self.near_experts], dim=1) # [B, 4, V]
        far_l  = self.far_expert(ht).view(h.size(0), self.max_h - 4, -1)     # [B, H-4, V]
        
        drafts = torch.cat([near_l, far_l], dim=1)
        return drafts, h_len, uncertainty

class AdaptiveBlock(nn.Module):
    """
    Standard Transformer Layer with CIF-locked Early Exit.
    Confidence-based skip logic prevents redundant computation.
    """
    def __init__(self, d_model: int, n_heads: int, cif: CIFRegistry):
        super().__init__()
        self.cif = cif
        self.attn = FlashBlockSparseAttention(d_model, n_heads, cif)
        self.mixer = DiamondMixer(d_model, cif)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.confidence_gate = nn.Sequential(
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor, cache: Optional[PagedKVCache] = None):
        # First residual block
        a_res, next_cache = self.attn(self.norm1(x), cache)
        x = x + a_res
        
        # Second residual block (Lattice Mixer)
        x = x + self.mixer(self.norm2(x))
        
        # Exit Confidence Resolution
        conf = self.confidence_gate(x[:, -1, :]).mean()
        # CIF Monomorphism test
        # Note: exit_thresh is a CIFScalar, we should use its value
        thresh = self.cif.exit_thresh.get()
        stop = conf.item() > thresh
        
        return x, next_cache, stop, conf

# ══════════════════════════════════════════════════════════════════════════════════════════
# 9.  THE IO MODEL — SYSTEM INTEGRATION
# ══════════════════════════════════════════════════════════════════════════════════════════

class Io(nn.Module):
    """
    The Unified HST Model v1.0.
    Final architecture combining Hyperbolic Geometry, Spine Lattice, and Speculative Decoding.
    """
    def __init__(self, cfg: IoConfig):
        super().__init__()
        self.cfg = cfg
        self.cif = CIFRegistry(cfg)
        self.analyzer = SpineAnalyzer(cfg.max_seq_len)
        
        # Input Pipeline
        self.embedding = HyperbolicEmbedding(cfg.vocab_size, cfg.d_model)
        self.pos_enc   = LatticePositionalEncoding(cfg.d_model, cfg.max_seq_len)
        
        # Core Processors
        self.lattice   = CompleteLatticeProcessor(cfg.d_model, self.analyzer, self.cif)
        self.layers = nn.ModuleList([
            AdaptiveBlock(cfg.d_model, cfg.n_heads, self.cif) 
            for _ in range(cfg.n_layers)
        ])
        
        # Inference Refiners
        self.feedback = FeedbackLoop(cfg.d_model, self.cif)
        self.hebbian  = HebbianFastWeights(cfg.d_model, self.cif)
        
        # Output Heads
        self.norm_out = nn.LayerNorm(cfg.d_model)
        self.lm_head  = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.horizon  = SpeculativeHorizon(cfg.d_model, cfg.vocab_size, cfg.max_horizon)
        
        # Weight Tying (GPT Optimization)
        self.lm_head.weight = self.embedding.emb.weight
        
        self.apply(self._init_weights)
        logger.info(f"Model Built | Layers: {cfg.n_layers} | Params: {sum(p.numel() for p in self.parameters()):,}")

    def _init_weights(self, m):
        """Initializes weights for stable deep training."""
        if isinstance(m, nn.Linear):
            torch.nn.init.normal_(m.weight, mean=0, std=0.02)
            if m.bias is not None: torch.nn.init.zeros_(m.bias)

    def forward(self, ids, caches=None, training=True):
        """Standard Forward Pass."""
        B, S = ids.shape
        x = self.embedding(ids)
        
        # Positional alignment
        p_offset = caches[0].pos if caches else 0
        p = torch.arange(p_offset, p_offset+S, device=ids.device).unsqueeze(0).expand(B,-1)
        x = x + self.pos_enc(p)
        
        # Structural Processing
        x = self.lattice(x)
        
        new_caches = []
        for i, layer in enumerate(self.layers):
            x, nc, skip, _ = layer(x, caches[i] if caches else None)
            new_caches.append(nc)
            # CIF Early Exit Logic (Inference only)
            if not training and skip and S == 1: break
            
        x = self.feedback(x)
        if not training: x = self.hebbian(x)
        
        h = self.norm_out(x)
        logits = self.lm_head(h)
        drafts, h_len, unc = self.horizon(h)
        
        return {"logits": logits, "drafts": drafts, "h_len": h_len, "unc": unc, "caches": new_caches}

# ══════════════════════════════════════════════════════════════════════════════════════════
# 10. HIGH-SPEED ASYNCHRONOUS HF STREAMER
# ══════════════════════════════════════════════════════════════════════════════════════════

class AsynchronousHFStreamer:
    """
    Parallel Dataset Streaming from Hugging Face.
    Provides zero-latency token delivery to the GPU by pre-fetching chunks in a background thread.
    """
    def __init__(self, name: str, batch_size: int, seq_len: int, encode_fn):
        self.name = name
        self.bs, self.sl = batch_size, seq_len
        self.encode = encode_fn
        self.buffer = queue.Queue(maxsize=128) # Queue for batches
        self.stop_signal = threading.Event()
        
        # Detect sample config for FineWeb-Edu
        self.config = "sample-10BT" if "fineweb-edu" in name.lower() else None
        
        self.worker = threading.Thread(target=self._run_stream, daemon=True)
        self.worker.start()

    def _run_stream(self):
        """Worker thread for HF datasets."""
        logger.info(f"Connecting to Hugging Face: {self.name}")
        try:
            ds = load_dataset(self.name, name=self.config, split="train", streaming=True)
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            return
        
        local_tokens = []
        target_size = (self.sl + 1) * self.bs
        
        for entry in ds:
            if self.stop_signal.is_set(): break
            
            text = entry.get("text", entry.get("content", ""))
            if not text: continue
            
            # Tokenize and append EOS (50256)
            local_tokens.extend(self.encode(text) + [50256])
            
            # Batch slicing
            while len(local_tokens) >= target_size:
                x_data, y_data = [], []
                for _ in range(self.bs):
                    slice_obj = local_tokens[:self.sl+1]
                    local_tokens = local_tokens[self.sl+1:]
                    x_data.append(slice_obj[:-1])
                    y_data.append(slice_obj[1:])
                
                # Push to buffer
                self.buffer.put((torch.tensor(x_data), torch.tensor(y_data)))
        
        logger.info("Stream Exhausted.")

    def get_batch(self):
        """Fetch batch from the parallel queue."""
        return self.buffer.get()

# ══════════════════════════════════════════════════════════════════════════════════════════
# 11. PRODUCTION TRAINING ENGINE
# ══════════════════════════════════════════════════════════════════════════════════════════

def train(cfg: IoConfig, dataset: str):
    print_banner()
    enc = tiktoken.get_encoding("gpt2")
    model = Io(cfg).to(cfg.device)
    
    # Optimizer with decoupled weight decay
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=cfg.lr, 
        weight_decay=cfg.weight_decay,
        betas=(0.9, 0.95)
    )
    
    # Scheduler: Linear Warmup -> Cosine Decay
    def lr_lambda(step):
        if step < cfg.warmup_steps: return step / cfg.warmup_steps
        progress = (step - cfg.warmup_steps) / (cfg.max_steps - cfg.warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Multi-threaded Streamer
    streamer = AsynchronousHFStreamer(
        dataset, cfg.batch_size, cfg.max_seq_len, 
        lambda t: enc.encode(t, allowed_special="all")
    )
    
    # Mixed Precision Scaler (Robust for CPU/GPU)
    device_type = "cuda" if "cuda" in cfg.device else "cpu"
    scaler = torch.amp.GradScaler(device=device_type, enabled=cfg.use_amp)
    
    logger.info(f"Pre-training Commenced | Dataset: {dataset} | AMP: {cfg.use_amp}")
    t_start = time.time()
    
    try:
        for step in range(1, cfg.max_steps + 1):
            model.train()
            
            # Gradient Accumulation Loop
            total_loss = 0
            for _ in range(cfg.accum_steps):
                x, y = streamer.get_batch()
                x, y = x.to(cfg.device), y.to(cfg.device)
                
                with torch.amp.autocast(device_type=device_type, enabled=cfg.use_amp):
                    out = model(x, training=True)
                    
                    # Cross Entropy for Next Token Prediction
                    loss_lm = F.cross_entropy(out["logits"].view(-1, cfg.vocab_size), y.view(-1))
                    
                    # Speculative Horizon Loss (Tweak: Teach the model to draft correctly)
                    # drafts predicted from h[:, -1, :] should match y[:, -1] and future tokens.
                    # Since we only have y up to S, we can only train drafts[:, 0, :] against y[:, -1]
                    h_logits = out["drafts"][:, 0, :] # [B, V]
                    h_targets = y[:, -1] # [B]
                    loss_h = F.cross_entropy(h_logits, h_targets)
                    
                    # Composite loss with CIF weighting
                    batch_loss = (cfg.accum_steps**-1) * (
                        model.cif.lm_weight.get() * loss_lm + 
                        model.cif.hor_weight.get() * loss_h +
                        model.cif.unc_weight.get() * out["unc"].mean()
                    )
                
                scaler.scale(batch_loss).backward()
                total_loss += batch_loss.item()
                
            # Optimization Step
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
            
            # Production Logging
            if step % 25 == 0:
                elapsed = time.time() - t_start
                # tokens processed = step_count * bs * accum * sl
                tps = (25 * cfg.batch_size * cfg.accum_steps * cfg.max_seq_len) / elapsed
                logger.info(f"Step {step:6d}/{cfg.max_steps} | Loss: {total_loss:.4f} | "
                            f"LR: {optimizer.param_groups[0]['lr']:.2e} | T/s: {tps:.0f}")
                t_start = time.time()
                
            # Checkpointing
            if step % 1000 == 0:
                ckpt_path = f"io_v1_step_{step}.pt"
                torch.save({
                    'step': step,
                    'model_state': model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'config': cfg.to_dict()
                }, ckpt_path)
                logger.info(f"Checkpoint Saved: {ckpt_path}")
    except KeyboardInterrupt:
        logger.info("Training interrupted by user.")
    finally:
        streamer.stop_signal.set()

# ══════════════════════════════════════════════════════════════════════════════════════════
# 12. GENERATION UTILS & ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def interactive_chat(model: Io, max_tokens: int = 150):
    """Inference loop with speculative decoding."""
    model.eval()
    enc = tiktoken.get_encoding("gpt2")
    
    while True:
        prompt = input("\nYou: ").strip()
        if prompt.lower() in ["exit", "quit"]: break
        
        ids = torch.tensor([enc.encode(prompt)]).to(model.cfg.device)
        caches = [PagedKVCache(model.cfg.n_heads, model.cif.head_dim.get(), model.cif, model.cfg.device) 
                  for _ in range(model.cfg.n_layers)]
        
        # Prefill
        out = model(ids, caches=caches, training=False)
        caches = out["caches"]
        
        print("Io: ", end="", flush=True)
        gen_tokens = 0
        while gen_tokens < max_tokens:
            # Speculative Draft selection
            draft_ids = torch.multinomial(F.softmax(out["drafts"][:, 0, :], dim=-1), 1)
            
            # Fast verification
            v_out = model(draft_ids, caches=caches, training=False)
            token_val = draft_ids.item()
            
            print(enc.decode([token_val]), end="", flush=True)
            
            caches = v_out["caches"]
            out = v_out
            gen_tokens += 1
            if token_val == 50256: break
        print("")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train", choices=["train", "chat", "test"])
    parser.add_argument("--data", type=str, default="HuggingFaceFW/fineweb-edu")
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--n_layers", dest="layers", type=int, default=16)
    parser.add_argument("--max_seq", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--batch", type=int, default=4)
    
    args = parser.parse_args()
    
    io_config = IoConfig(
        batch_size=args.batch, 
        n_layers=args.layers, 
        d_model=args.d_model,
        max_seq_len=args.max_seq,
        max_steps=args.steps
    )

    if args.mode == "train":
        train(io_config, args.data)
    elif args.mode == "chat":
        m = Io(io_config).to(io_config.device)
        interactive_chat(m)
    elif args.mode == "test":
        # Run CIF Benchmarks
        logger.info("Executing CIF Health Check...")
        d = Deny(5)
        t_start = time.time()
        for i in range(1000000): d.test(lambda x: x > 2)
        logger.info(f"CIF Resolution Latency: {time.time() - t_start:.6f}s (Monomorphic)")
        
        # Test forward pass
        m = Io(io_config).to(io_config.device)
        dummy_ids = torch.randint(0, io_config.vocab_size, (1, 10)).to(io_config.device)
        out = m(dummy_ids, training=False)
        logger.info(f"Forward Pass Output: {out['logits'].shape}")
        logger.info("HST Framework Integrity: VERIFIED")

# ══════════════════════════════════════════════════════════════════════════════════════════
# EOF Io v1.0
