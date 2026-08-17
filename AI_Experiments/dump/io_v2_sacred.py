import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import time
import random
from typing import Optional, List, Tuple, Dict, Callable, Any
from abc import ABC, abstractmethod

# ==============================================================================
# 1. PELL-LUCAS TIME SPINE (Infinite Context Encoding)
# ==============================================================================

class PellLucasTimeSpine(nn.Module):
    """
    Encodes both absolute position and lattice hierarchy using Pell-Lucas sequences.
    P_n = 2*P_{n-1} + P_{n-2}
    """
    def __init__(self, d_model, max_seq_len=8192):
        super().__init__()
        self.d_model = d_model
        
        # Standard sinusoidal for absolute position (half dim)
        self.absolute_pe = self._get_sinusoidal_encoding(max_seq_len, d_model // 2)
        
        # Lattice-based encoding (half dim)
        spine = self._generate_pell_spine(max_seq_len)
        self.register_buffer('spine', torch.tensor(spine))
        
        self.lattice_encoder = nn.Sequential(
            nn.Linear(3, d_model // 2),  # left_dist, right_dist, level
            nn.LayerNorm(d_model // 2),
            nn.GELU()
        )
    
    def forward(self, positions):
        B, S = positions.shape
        device = positions.device
        
        # Absolute encoding
        abs_enc = self.abs_pe[positions]
        
        # Lattice encoding
        # Vectorized lattice distance calculation
        pos_flat = positions.reshape(-1)
        
        # For each position, find nearest spine points
        # Using searchsorted for efficiency
        idx = torch.searchsorted(self.spine, pos_flat)
        idx = torch.clamp(idx, 1, len(self.spine) - 1)
        
        left_val = self.spine[idx - 1]
        right_val = self.spine[idx]
        
        left_dist = pos_flat - left_val
        right_dist = right_val - pos_flat
        level = idx.float()
        
        lattice_features = torch.stack([left_dist.float(), right_dist.float(), level], dim=-1)
        lat_enc = self.lattice_encoder(lattice_features)
        lat_enc = lat_enc.view(B, S, -1)
        
        return torch.cat([abs_enc, lat_enc], dim=-1)
    
    @staticmethod
    def _generate_pell_spine(max_len):
        # S_n = 2*S_{n-1} + S_{n-2}
        spine = [0, 2, 5] # Initial Pell-like
        while spine[-1] < max_len:
            next_val = 2 * spine[-1] + spine[-2]
            spine.append(next_val)
        return spine
    
    def _get_sinusoidal_encoding(self, max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('abs_pe', pe)
        return pe

# ==============================================================================
# 2. DIAMOND MIXER (Lossless Logic)
# ==============================================================================

class DiamondMixer(nn.Module):
    """
    Topology:
      x, y = Split(Input)
      Z = x + y (Synthesis/Context)
      W = y - x (Analysis/Detail)
      Output = Merge(Z, W)
    """
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.split_proj = nn.Linear(d_model, d_model * 2)
        self.z_process = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU())
        self.w_process = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU())
        self.merge_proj = nn.Linear(d_model * 2, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, u):
        xy = self.split_proj(u)
        x, y = xy.chunk(2, dim=-1)
        z = x + y  # Synthesis
        w = y - x  # Analysis
        z_prime = self.z_process(z)
        w_prime = self.w_process(w)
        out = self.merge_proj(torch.cat([z_prime, w_prime], dim=-1))
        return self.norm(u + out)

# ==============================================================================
# 3. HOLOGRAPHIC LATTICE (Interference Field)
# ==============================================================================

class HolographicLattice(nn.Module):
    """
    Multi-level interference based on the Pell spine.
    Calculates interactions between nodes at different hierarchical levels.
    """
    def __init__(self, d_model, max_seq_len=8192):
        super().__init__()
        self.d_model = d_model
        # Generate spine for level reference
        spine = [0, 2, 5]
        while spine[-1] < max_seq_len:
            spine.append(2 * spine[-1] + spine[-2])
        self.register_buffer('spine', torch.tensor(spine))
        
        self.interference_proj = nn.Linear(d_model, d_model)
        self.level_attn = nn.MultiheadAttention(d_model, 8, batch_first=True)
        self.fusion = nn.Linear(d_model * 2, d_model)

    def forward(self, x):
        B, S, D = x.shape
        # Identify spine points in current sequence
        seq_spine = self.spine[self.spine < S]
        if len(seq_spine) < 2:
            return x
            
        # Extract spine node features
        spine_features = x[:, seq_spine, :] # [B, spine_len, D]
        
        # Global interference field
        interference, _ = self.level_attn(x, spine_features, spine_features)
        
        # Fuse with original signal
        combined = torch.cat([x, interference], dim=-1)
        return self.fusion(combined)

# ==============================================================================
# 4. HEBBIAN FAST WEIGHTS (Plasticity)
# ==============================================================================

class HebbianFastWeights(nn.Module):
    """
    Plasticity layer that maintains a transient memory state.
    """
    def __init__(self, d_model, lambda_decay=0.9):
        super().__init__()
        self.d_model = d_model
        self.lambda_decay = lambda_decay
        self.qkv = nn.Linear(d_model, d_model * 3, bias=False)
        self.norm = nn.LayerNorm(d_model)
        
        # Persistent memory for plasticity across forward passes during inference
        self.register_buffer('fast_weights', torch.zeros(1, d_model, d_model))

    def reset_plasticity(self):
        self.fast_weights.zero_()

    def forward(self, x, training=True):
        B, S, D = x.shape
        qkv = self.qkv(x).reshape(B, S, 3, D).permute(2, 0, 1, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Update fast weights (Hebbian step)
        # Using a simplified version for integration
        # dW = k^T * v
        current_update = torch.einsum('bsd,bse->bde', k, v) / math.sqrt(D)
        
        if not training:
            # Add to persistent memory during inference
            # Ensure shape matches batch
            if self.fast_weights.size(0) != B:
                # Use data update to avoid breaking buffer registration
                self.fast_weights.data = self.fast_weights.data.expand(B, -1, -1).contiguous()
            
            # In-place update to maintain buffer persistence
            self.fast_weights.data.mul_(self.lambda_decay).add_(current_update, alpha=(1 - self.lambda_decay))
            fw = self.fast_weights
        else:
            # Local update only for training to keep gradients sane
            fw = current_update
            
        # Retrieve from fast weights
        out = torch.einsum('bsd,bde->bse', q, fw)
        
        # Dynamic gate
        gate = torch.sigmoid((q * k).sum(dim=-1, keepdim=True))
        return self.norm(x + out * gate)

# ==============================================================================
# 5. CHAOS LOGIC & ERROR NETWORKS
# ==============================================================================

class ChaosRefinement(nn.Module):
    """
    Iterative refinement with Chaos Logic.
    """
    def __init__(self, d_model, iterations=2):
        super().__init__()
        self.iterations = iterations
        self.refiner = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model)
        )
        self.chaos_scale = nn.Parameter(torch.tensor(0.01))

    def forward(self, x, intensity_mult=1.0):
        h = x
        for _ in range(self.iterations):
            # Inject Chaos
            chaos = torch.randn_like(h) * self.chaos_scale * intensity_mult
            h = h + chaos
            # Creation (Refinement)
            h = h + self.refiner(h)
        return h

class ErrorSupervisor:
    def __init__(self):
        self.chaos_mult = 1.0
        self.history = []

    def report_performance(self, loss):
        self.history.append(loss)
        if len(self.history) > 10:
            avg_recent = sum(self.history[-5:]) / 5
            avg_older = sum(self.history[-10:-5]) / 5
            
            if avg_recent > avg_older:
                # Loss increasing, system unstable, reduce chaos
                self.chaos_mult = max(0.1, self.chaos_mult * 0.8)
            else:
                # System stable, can increase chaos for more "creativity"
                self.chaos_mult = min(2.0, self.chaos_mult * 1.05)
        return self.chaos_mult

# ==============================================================================
# Io v2 SACRED (FULL CRYSTALLINE v8.2)
# ==============================================================================

class HyperbolicEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, curvature=1.0):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.c = curvature
        
    def forward(self, x):
        e = self.embed(x)
        norm = e.norm(dim=-1, keepdim=True)
        max_norm = (1 - 1e-3) / math.sqrt(self.c)
        return e * (torch.clamp(max_norm / norm, max=1.0))

class IoSacred(nn.Module):
    def __init__(self, vocab_size, d_model=768, n_heads=12, n_layers=12, max_seq_len=8192):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        self.embedding = HyperbolicEmbedding(vocab_size, d_model)
        self.time_spine = PellLucasTimeSpine(d_model, max_seq_len)
        self.plasticity = HebbianFastWeights(d_model)
        
        # Main Crystalline Stack
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'attn': nn.MultiheadAttention(d_model, n_heads, batch_first=True),
                'mixer': DiamondMixer(d_model),
                'norm1': nn.LayerNorm(d_model),
                'norm2': nn.LayerNorm(d_model)
            }) for _ in range(n_layers)
        ])
        
        self.lattice = HolographicLattice(d_model, max_seq_len)
        self.chaos = ChaosRefinement(d_model)
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embedding.embed.weight
        
        self.supervisor = ErrorSupervisor()

    def forward(self, input_ids, cache=None, chaos_mult=1.0):
        B, S = input_ids.shape
        device = input_ids.device
        
        # 1. Embeddings & Pell-Lucas Time Spine
        positions = torch.arange(0, S, device=device).unsqueeze(0).expand(B, -1)
        x = self.embedding(input_ids) + self.time_spine(positions)
        
        # 2. Hebbian Plasticity
        x = self.plasticity(x, training=self.training)
        
        # 3. Crystalline Stack
        for layer in self.layers:
            # Attention
            attn_out, _ = layer['attn'](layer['norm1'](x), x, x, need_weights=False)
            x = x + attn_out
            # Diamond Mixer
            x = x + layer['mixer'](layer['norm2'](x))
            
        # 4. Holographic Lattice
        x = self.lattice(x)
        
        # 5. Chaos Refinement
        x = self.chaos(x, intensity_mult=chaos_mult)
        
        # 6. Output
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        return {"logits": logits}

    @torch.no_grad()
    def generate(self, prompt, max_new_tokens, temperature=1.0, top_k=50):
        self.eval()
        current_ids = prompt
        self.plasticity.reset_plasticity()
        
        # In generation, we use the supervisor's current mult
        chaos_mult = self.supervisor.chaos_mult
        
        for _ in range(max_new_tokens):
            # Autoregressive generation (full context for simplicity in this implementation)
            outputs = self.forward(current_ids, chaos_mult=chaos_mult)
            logits = outputs["logits"][:, -1, :]
            
            if top_k > 0:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, -1].unsqueeze(-1)] = float('-inf')
            
            probs = F.softmax(logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, 1)
            current_ids = torch.cat([current_ids, next_token], dim=1)
            
            if next_token == 0: break
                
        return current_ids

if __name__ == "__main__":
    model = IoSacred(vocab_size=50257)
    print(f"Io v2 Sacred (v8.2 Crystalline) Initialized.")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    x = torch.randint(0, 50257, (1, 10))
    out = model(x)
    print(f"Forward pass output: {out['logits'].shape}")
