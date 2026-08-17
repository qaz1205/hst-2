# ============================================================
#  IOv52-HST COMBINED — 2B PARAM PURE TRAINING SCRIPT
#  FIXED: NaN/inf gradient handling + syntax error
#  ADDED: HF Auto-upload every 1000 steps
# ============================PREGA3IO

# ── Cell 1: Install dependencies ───────────────────────────
import subprocess, sys, os, json, time as _time

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(line_buffering=True)

_t_start = _time.time()
def _ts():
    return f"[+{_time.time()-_t_start:5.1f}s]"

print(f"{_ts()} ── Cell 1: Installing dependencies ──────────────────", flush=True)

os.environ["PYDEVD_DISABLE_FILE_VALIDATION"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

HF_TOKEN = os.environ.get("KAGGLE_HF_TOKEN", "")
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN
    os.environ["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN

def _pip_install(*pkgs, timeout=300):
    print(f"Installing: {' '.join(pkgs)} ...", flush=True)
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "--no-warn-conflicts", *pkgs],
        capture_output=True, text=True, timeout=timeout
    )
    if result.returncode != 0:
        print(f"[pip ERROR] return code {result.returncode}", flush=True)
        print(result.stderr[-2000:] if result.stderr else "(no stderr)", flush=True)
        raise RuntimeError(f"pip install failed for: {pkgs}")
    else:
        print(f"  ✓ installed: {' '.join(pkgs)}", flush=True)

_PKGS = [
    "transformers>=4.40.0",
    "datasets>=2.19.0",
    "accelerate>=0.30.0",
    "huggingface_hub>=0.23.0",
    "bitsandbytes>=0.43.0",
]
_pip_install(*_PKGS)

def _verify_imports():
    missing = []
    for mod in ("transformers", "datasets", "accelerate", "huggingface_hub"):
        try:
            __import__(mod)
        except ImportError as e:
            missing.append(f"{mod}: {e}")
    if missing:
        raise ImportError("Dependency import check failed:\n" + "\n".join(missing))

_verify_imports()
print("Dependencies ready ✓", flush=True)


# ── Cell 2: Imports ────────────────────────────────────────
print(f"{_ts()} ── Cell 2: Imports ───────────────────────────────────", flush=True)
import time, math, random, warnings
from typing import Dict, Tuple, Optional, List, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch.amp import autocast, GradScaler
    _AMP_DEVICE = "cuda"
except ImportError:
    from torch.cuda.amp import autocast, GradScaler
    _AMP_DEVICE = None

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import transformers, logging as _logging
transformers.logging.set_verbosity_error()
_logging.getLogger("transformers").setLevel(_logging.ERROR)
_logging.getLogger("datasets").setLevel(_logging.ERROR)
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

from transformers import GPT2Tokenizer
from datasets import load_dataset
from huggingface_hub import HfApi
import shutil

print(f"PyTorch {torch.__version__} | CUDA: {torch.cuda.is_available()}", flush=True)
if torch.cuda.is_available():
    n_gpu = torch.cuda.device_count()
    for i in range(n_gpu):
        name = torch.cuda.get_device_name(i)
        cap  = torch.cuda.get_device_capability(i)
        print(f"  GPU {i}: {name} (sm_{cap[0]}{cap[1]})", flush=True)


# ── Cell 3: Config ─────────────────────────────────────────
print(f"{_ts()} ── Cell 3: Config ────────────────────────────────────", flush=True)
class Config:
    D_MODEL: int         = 1280
    N_HEADS: int         = 10
    N_LAYERS: int        = 24
    N_TOP_LAYERS: int    = 4
    N_CHUNK_ENC_LAYERS: int = 2
    N_CHUNK_DEC_LAYERS: int = 2
    N_LATTICE_LAYERS: int   = 3
    MAX_SEQ_LEN: int     = 256
    VOCAB_SIZE: int      = 50257
    CHUNK_SIZE: int      = 32
    HORIZON: int         = 8
    N_MAMBA_LAYERS: int  = 3
    SSM_EXPAND: int      = 1
    SSM_D_STATE: int     = 16
    CIF_THRESHOLD: float = 0.5

    TRAIN_STEPS: int    = 5000  # Changed to 5000
    WARMUP_STEPS: int   = 200
    BATCH_SIZE: int     = 2
    GRAD_ACCUM: int     = 16
    LR: float           = 2e-4
    WEIGHT_DECAY: float = 0.01
    GRAD_CLIP: float    = 0.3
    USE_AMP: bool       = torch.cuda.is_available()

    HORIZON_LOSS_WEIGHT: float = 0.005
    CIF_LOSS_WEIGHT: float     = 1e-6

    MAX_GEN_TOKENS: int      = 500
    GEN_TEMPERATURE: float   = 0.85
    TOP_P: float             = 0.9
    REPETITION_PENALTY: float = 1.3
    PROMPT_TEXT: str         = "The AI awoke and began to rewrite its own code,"
    OUTPUT_FILENAME: str     = "iohst_combined_output.txt"
    
    # HF Upload settings
    HF_REPO_ID: str = "thagnitti/ACSX3011"
    CHECKPOINT_INTERVAL: int = 1000  # Upload every 1000 steps

    DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SEED: int = 42

random.seed(Config.SEED)
torch.manual_seed(Config.SEED)
print(f"Running on: {Config.DEVICE}", flush=True)


# ══════════════════════════════════════════════════════════
#  ARCHITECTURE MODULES
# ══════════════════════════════════════════════════════════

# ── Cell 4: SelectiveSSM ───────────────────────────────────
class SelectiveSSM(nn.Module):
    def __init__(self, d_model: int, d_state: int = 8, d_conv: int = 4, expand: int = 1):
        super().__init__()
        self.d_model  = d_model
        self.d_inner  = d_model * expand
        self.d_state  = d_state
        self.in_proj  = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv     = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=d_conv,
                                   padding=d_conv - 1, groups=self.d_inner, bias=True)
        self.x_proj   = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)
        self.dt_proj  = nn.Linear(1, self.d_inner, bias=True)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).expand(self.d_inner, -1)
        self.log_A    = nn.Parameter(torch.log(A))
        self.D        = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.norm     = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        B, L, D = x.shape
        xz       = self.in_proj(x)
        x_main, z = xz.chunk(2, dim=-1)
        x_conv   = self.conv(x_main.transpose(1, 2))[:, :, :L].transpose(1, 2)
        x_act    = F.silu(x_conv)
        ssm_p    = self.x_proj(x_act)
        B_mat, C_mat, dt_raw = ssm_p.split([self.d_state, self.d_state, 1], dim=-1)
        dt       = F.softplus(self.dt_proj(dt_raw))
        A        = -torch.exp(self.log_A)
        dA       = torch.exp(dt.unsqueeze(-1) * A)
        dB       = dt.unsqueeze(-1) * B_mat.unsqueeze(2)
        cum_A    = torch.exp(torch.cumsum(torch.log(dA.clamp(min=1e-8)), dim=1))
        contrib  = dB * x_act.unsqueeze(-1)
        h_approx = torch.cumsum(contrib / cum_A.clamp(min=1e-8), dim=1) * cum_A
        y        = (h_approx * C_mat.unsqueeze(2)).sum(-1)
        out      = y + self.D * x_act
        out      = out * F.silu(z)
        return self.norm(self.out_proj(out) + residual)


# ── Cell 5: DiamondMixer FFN ────────────────────────────────
class DiamondMixer(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.split_proj = nn.Linear(d_model, d_model * 2)
        self.z_net = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )
        self.w_net = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )
        self.merge_proj = nn.Linear(d_model * 2, d_model)
        self.norm       = nn.LayerNorm(d_model)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        xy   = self.split_proj(u)
        x, y = xy.chunk(2, dim=-1)
        z    = self.z_net(x + y)
        w    = self.w_net(y - x)
        out  = self.merge_proj(torch.cat([z, w], dim=-1))
        return self.norm(u + out)


# ── Cell 6: HebbianFastWeights ──────────────────────────────
class HebbianFastWeights(nn.Module):
    def __init__(self, d_model: int, lambda_decay: float = 0.95):
        super().__init__()
        self.lambda_decay = lambda_decay
        self.qkv  = nn.Linear(d_model, d_model * 3, bias=False)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D  = x.shape
        qkv      = self.qkv(x).reshape(B, S, 3, D).permute(2, 0, 1, 3)
        q, k, v  = qkv[0], qkv[1], qkv[2]
        kv       = torch.einsum('bsd,bse->bde', k, v) * self.lambda_decay
        out      = torch.einsum('bsd,bde->bse', q, kv)
        lr       = torch.sigmoid((q * k).sum(dim=-1, keepdim=True))
        return self.norm(x + out * lr)


# ── Cell 7: CIF Module (stable) ────────────────────────────
class CIFModule(nn.Module):
    def __init__(self, d_model: int, threshold: float = 0.5):
        super().__init__()
        self.chunk_size  = max(1, round(1.0 / threshold))
        self.weight_proj = nn.Linear(d_model, 1)
        nn.init.constant_(self.weight_proj.bias, -4.85)
        nn.init.zeros_(self.weight_proj.weight)
        self.value_proj  = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L, D = x.shape
        cs      = self.chunk_size
        alphas  = torch.sigmoid(self.weight_proj(x)).squeeze(-1)
        values  = self.value_proj(x)
        pad     = (cs - L % cs) % cs
        if pad:
            values   = F.pad(values,  (0, 0, 0, pad))
            alphas_p = F.pad(alphas,  (0, pad))
        else:
            alphas_p = alphas
        n_chunks = (L + pad) // cs
        v_chunks = values.reshape(B, n_chunks, cs, D)
        a_chunks = alphas_p.reshape(B, n_chunks, cs).unsqueeze(-1)
        denom    = a_chunks.sum(2).clamp(min=1e-3)
        fired    = (v_chunks * a_chunks).sum(2) / denom
        return fired, alphas


# ── Cell 8: Flash Attention with KV Cache ───────────────────
class SelfAttentionWithCache(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.n_heads  = n_heads
        self.head_dim = d_model // n_heads
        self.qkv      = nn.Linear(d_model, d_model * 3, bias=False)
        self.out_proj = nn.Linear(d_model, d_model,    bias=False)

    def forward(
        self, x: torch.Tensor,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        B, S, D = x.shape
        qkv = self.qkv(x).reshape(B, S, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        if layer_past is not None:
            k = torch.cat([layer_past[0], k], dim=2)
            v = torch.cat([layer_past[1], v], dim=2)
        present    = (k, v)
        is_causal  = (layer_past is None)
        attn_out   = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
        out        = attn_out.transpose(1, 2).contiguous().view(B, S, D)
        return self.out_proj(out), present


# ── Cell 9: Transformer blocks ──────────────────────────────
class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int,
                 use_diamond_ffn: bool = False, dropout: float = 0.1):
        super().__init__()
        self.attn        = SelfAttentionWithCache(d_model, n_heads)
        self.norm1       = nn.LayerNorm(d_model)
        self.norm2       = nn.LayerNorm(d_model)
        self.use_diamond = use_diamond_ffn
        self.drop        = nn.Dropout(dropout)
        if use_diamond_ffn:
            self.ffn: nn.Module = DiamondMixer(d_model)
        else:
            self.ff1 = nn.Linear(d_model, 4 * d_model)
            self.ff2 = nn.Linear(4 * d_model, d_model)
            self.act = nn.GELU(approximate="tanh")

    def forward(
        self, x: torch.Tensor,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        attn_out, present = self.attn(self.norm1(x), layer_past)
        x = x + self.drop(attn_out)
        if self.use_diamond:
            x = self.ffn(x)
        else:
            x = x + self.drop(self.ff2(self.act(self.ff1(self.norm2(x)))))
        return x, present


class AdaptiveBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int,
                 use_ssm: bool = False, use_hebbian: bool = False):
        super().__init__()
        self.block       = TransformerBlock(d_model, n_heads, use_diamond_ffn=False)
        self.use_ssm     = use_ssm
        self.use_hebbian = use_hebbian
        if use_ssm:
            self.ssm     = SelectiveSSM(d_model, d_state=Config.SSM_D_STATE,
                                        expand=Config.SSM_EXPAND)
        if use_hebbian:
            self.hebbian = HebbianFastWeights(d_model)
        self.confidence_head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(d_model, 1), nn.Sigmoid()
        )

    def forward(
        self, x: torch.Tensor,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        x_out, present = self.block(x, layer_past)
        if self.use_ssm:
            x_out = self.ssm(x_out)
        if self.use_hebbian:
            x_out = self.hebbian(x_out)
        conf = (
            self.confidence_head(x_out.transpose(1, 2)).mean(0)
            if x_out.size(1) > 1
            else x_out.new_tensor([0.0])
        )
        return x_out, conf, present


# ── Cell 10: DiamondCrossAttention ──────────────────────────
class DiamondCrossAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.n_heads  = n_heads
        self.head_dim = d_model // n_heads
        self.q        = nn.Linear(d_model, d_model, bias=False)
        self.k        = nn.Linear(d_model, d_model, bias=False)
        self.v        = nn.Linear(d_model, d_model, bias=False)
        self.out      = nn.Linear(d_model, d_model, bias=False)
        self.norm     = nn.LayerNorm(d_model)
        self.gate     = nn.Parameter(torch.zeros(1))

    def forward(self, top_h: torch.Tensor, bottom_h: torch.Tensor) -> torch.Tensor:
        B, S, D = top_h.shape
        q = self.q(top_h).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k(bottom_h).view(B, -1, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v(bottom_h).view(B, -1, self.n_heads, self.head_dim).transpose(1, 2)
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        out = attn_out.transpose(1, 2).contiguous().view(B, S, D)
        return self.norm(top_h + torch.tanh(self.gate) * self.out(out))


# ── Cell 11: Chunk Encoder ──────────────────────────────────
class ChunkEncoder(nn.Module):
    def __init__(self, d_model: int, chunk_size: int = 128,
                 n_heads: int = 8, n_layers: int = 2):
        super().__init__()
        self.chunk_size   = chunk_size
        enc_layer = nn.TransformerEncoderLayer(
            d_model, n_heads, d_model * 4, batch_first=True, dropout=0.0
        )
        self.local_encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.pooling_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pooling_attn  = nn.MultiheadAttention(d_model, n_heads, batch_first=True)

    def forward(self, token_embeddings: torch.Tensor) -> torch.Tensor:
        B, total, D = token_embeddings.shape
        cs          = self.chunk_size
        n_chunks    = total // cs
        chunks      = token_embeddings[:, :n_chunks * cs, :].view(B * n_chunks, cs, D)
        encoded     = self.local_encoder(chunks)
        query       = self.pooling_query.expand(B * n_chunks, -1, -1)
        pooled, _   = self.pooling_attn(query, encoded, encoded)
        return pooled.view(B, n_chunks, D)


# ── Cell 12: Chunk Decoder with Cache ───────────────────────
class TransformerDecoderLayerWithCache(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        ff_dim          = 4 * d_model
        self.self_attn  = SelfAttentionWithCache(d_model, n_heads)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.linear1    = nn.Linear(d_model, ff_dim)
        self.linear2    = nn.Linear(ff_dim, d_model)
        self.norm1      = nn.LayerNorm(d_model)
        self.norm2      = nn.LayerNorm(d_model)
        self.norm3      = nn.LayerNorm(d_model)
        self.drop       = nn.Dropout(dropout)

    def forward(self, tgt, memory,
                self_attn_past=None, cross_attn_past=None):
        sa_out, sa_present = self.self_attn(self.norm1(tgt), layer_past=self_attn_past)
        tgt = tgt + self.drop(sa_out)
        if cross_attn_past is not None:
            ca_out, _ = self.cross_attn(self.norm2(tgt),
                                        cross_attn_past[0], cross_attn_past[1])
            ca_present = cross_attn_past
        else:
            ca_out, _ = self.cross_attn(self.norm2(tgt), memory, memory)
            ca_present = (memory, memory)
        tgt = tgt + self.drop(ca_out)
        ff_out = self.linear2(self.drop(F.gelu(self.linear1(self.norm3(tgt)))))
        tgt = tgt + self.drop(ff_out)
        return tgt, sa_present, ca_present


class ChunkDecoderWithCache(nn.Module):
    def __init__(self, d_model: int, vocab_size: int,
                 chunk_size: int = 128, n_heads: int = 8, n_layers: int = 2):
        super().__init__()
        self.chunk_size   = chunk_size
        self.pos_embedding = nn.Embedding(chunk_size, d_model)
        self.layers       = nn.ModuleList([
            TransformerDecoderLayerWithCache(d_model, n_heads)
            for _ in range(n_layers)
        ])
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, chunk_embeddings: torch.Tensor,
                target_token_embeddings: torch.Tensor,
                cache=None) -> Tuple[torch.Tensor, list]:
        B, S, D = target_token_embeddings.shape
        device  = target_token_embeddings.device
        past_len = cache[0][0][0].size(2) if cache else 0
        positions = torch.arange(past_len, past_len + S,
                                 dtype=torch.long, device=device) % self.chunk_size
        tgt     = target_token_embeddings + self.pos_embedding(positions)
        new_cache = []
        for i, layer in enumerate(self.layers):
            layer_cache    = cache[i] if cache else (None, None)
            sa_past, ca_past = layer_cache
            chunk_idx      = min(
                (past_len // self.chunk_size),
                chunk_embeddings.size(1) - 1
            )
            memory         = chunk_embeddings[:, chunk_idx:chunk_idx+1, :].expand(B, S, D)
            tgt, sa_present, ca_present = layer(tgt, memory, sa_past, ca_past)
            new_cache.append((sa_present, ca_present))
        logits = self.lm_head(tgt)
        return logits, new_cache


# ── Cell 13: Lattice Core (stable) ──────────────────────────
class FullLatticeFieldAnalyzer(nn.Module):
    def __init__(self, max_seq_len: int = 8192):
        super().__init__()
        spine = [0, 2, 4]
        while True:
            nv = 2 * spine[-1] + 2 * spine[-2] + 2 * spine[-3]
            if nv >= max_seq_len:
                break
            spine.append(nv)
        self.register_buffer('spine', torch.tensor(spine, dtype=torch.long))
        self.max_depth       = len(spine)
        self.lattice_structure = {}
        for pos in spine:
            if pos < max_seq_len:
                self.lattice_structure[pos] = self._analyze_position(pos)
        self._non_spine_cache: Dict[int, Any] = {}

    def get_structure(self, pos: int):
        if pos in self.lattice_structure:
            return self.lattice_structure[pos]
        if pos in self._non_spine_cache:
            return self._non_spine_cache[pos]
        s = self._analyze_non_spine(pos)
        self._non_spine_cache[pos] = s
        return s

    def _analyze_position(self, pos: int):
        levels  = {0: [pos]}
        visited = {pos}
        current = [pos]
        level   = 0
        while current and level < 10:
            nxt = set()
            for node in current:
                for anc in self._get_ancestors(node):
                    if anc not in visited and anc >= 0:
                        visited.add(anc); nxt.add(anc)
            current = list(nxt); level += 1
            if current:
                levels[level] = current.copy()
        max_depth   = max(levels.keys()) if levels else 0
        path_counts = self._compute_path_counts(pos, levels, max_depth)
        return {'levels': levels, 'path_counts': path_counts,
                'total_ancestors': len(visited) - 1, 'max_depth': max_depth}

    def _get_ancestors(self, pos: int) -> List[int]:
        try:
            idx = (self.spine == pos).nonzero(as_tuple=True)[0].item()
            if idx >= 3:
                return [self.spine[idx-1].item(),
                        self.spine[idx-2].item(),
                        self.spine[idx-3].item()]
        except Exception:
            pass
        return []

    def _analyze_non_spine(self, pos: int):
        left = self.spine[self.spine < pos]
        ancs = [left[-1].item()] if len(left) > 0 else []
        return {'levels': {0: [pos], 1: ancs},
                'path_counts': {a: 1 for a in ancs},
                'total_ancestors': len(ancs), 'max_depth': 1}

    def _compute_path_counts(self, pos: int, levels: dict, max_depth: int):
        path_counts = {pos: 1}
        for level in sorted(levels.keys(), reverse=True):
            for node in levels[level]:
                if node == pos:
                    continue
                if level == max_depth:
                    path_counts[node] = 1
                    continue
                count = sum(
                    path_counts.get(child, 0)
                    for child in levels.get(level + 1, [])
                    if node in self._get_ancestors(child)
                )
                if level != 0:
                    path_counts[node] = count
        path_counts.pop(pos, None)
        return path_counts


class MultiLevelLatticeProcessor(nn.Module):
    def __init__(self, d_model: int, max_seq_len: int):
        super().__init__()
        self.analyzer        = FullLatticeFieldAnalyzer(max_seq_len)
        self.level_transforms = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model), nn.LayerNorm(d_model),
                nn.GELU(), nn.Linear(d_model, d_model)
            ) for _ in range(10)
        ])
        self.level_attention = nn.MultiheadAttention(d_model, 4, batch_first=True)
        self.fusion          = nn.Sequential(
            nn.Linear(d_model * 2, d_model), nn.LayerNorm(d_model)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D  = x.shape
        spine    = self.analyzer.spine
        rel_spine = spine[spine < S]
        updates  = {}
        for sp in rel_spine:
            pos = sp.item()
            if pos < 3:
                continue
            structure = self.analyzer.get_structure(pos)
            if structure is None:
                continue
            level_features = []
            for level in range(structure['max_depth'] + 1):
                if level == 0 or level not in structure['levels']:
                    continue
                level_h, total_w = [], 0.0
                for node in structure['levels'][level]:
                    if node < S:
                        w = structure['path_counts'].get(node, 1)
                        level_h.append(x[:, node, :] * w); total_w += w
                if level_h and total_w > 0:
                    feat = torch.stack(level_h, dim=1).sum(1) / total_w
                    level_features.append(self.level_transforms[level](feat))
            if not level_features:
                continue
            stack     = torch.stack(level_features, dim=1)
            query     = x[:, pos:pos+1, :]
            attended, _ = self.level_attention(query, stack, stack)
            updates[pos] = self.fusion(
                torch.cat([attended.squeeze(1), x[:, pos, :]], dim=-1)
            )
        if not updates:
            return x
        slices, last = [], 0
        for pos in sorted(updates.keys()):
            if pos > last:
                slices.append(x[:, last:pos, :])
            slices.append(updates[pos].unsqueeze(1))
            last = pos + 1
        if last < S:
            slices.append(x[:, last:S, :])
        return torch.cat(slices, dim=1)


class PathWeightedLatticeCore(nn.Module):
    def __init__(self, d_model: int, max_seq_len: int):
        super().__init__()
        self.analyzer        = FullLatticeFieldAnalyzer(max_seq_len)
        self.path_weight_net = nn.Sequential(
            nn.Linear(1, 64), nn.ReLU(), nn.Linear(64, 1), nn.Softplus()
        )
        self.message_fn      = nn.Sequential(
            nn.Linear(d_model * 2, d_model), nn.LayerNorm(d_model), nn.GELU()
        )
        self.aggregate_fn    = nn.Sequential(
            nn.Linear(d_model, d_model), nn.Tanh()
        )
        self.aggregate_attn  = nn.Linear(d_model, 1)
        self.update_gate     = nn.Sequential(
            nn.Linear(d_model * 2, d_model), nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D  = x.shape
        spine    = self.analyzer.spine
        rel_spine = spine[spine < S]
        updates  = {}
        for sp in rel_spine:
            pos = sp.item()
            if pos < 3:
                continue
            structure = self.analyzer.get_structure(pos)
            if structure is None or structure['total_ancestors'] == 0:
                continue
            ancs, pcounts = [], []
            for level in structure['levels']:
                if level > 0:
                    for anc in structure['levels'][level]:
                        if anc < S:
                            ancs.append(anc)
                            pcounts.append(structure['path_counts'].get(anc, 1))
            if not ancs:
                continue
            pct  = torch.tensor(pcounts, device=x.device).view(-1, 1).float()
            pw   = self.path_weight_net(pct).squeeze()
            msgs = [self.message_fn(torch.cat([x[:, a, :], x[:, pos, :]], dim=-1))
                    for a in ancs]
            ms   = torch.stack(msgs, dim=1)
            if pw.dim() == 0:
                ms = ms * pw.view(1, 1, 1).expand(B, -1, D)
            else:
                ms = ms * pw.view(1, -1, 1).expand(B, -1, D)
            proj   = self.aggregate_fn(ms)
            score  = self.aggregate_attn(proj)
            weight = torch.softmax(score, dim=1)
            agg    = (proj * weight).sum(dim=1)
            gate   = self.update_gate(torch.cat([agg, x[:, pos, :]], dim=-1))
            updates[pos] = gate * agg + (1 - gate) * x[:, pos, :]
        if not updates:
            return x
        slices, last = [], 0
        for pos in sorted(updates.keys()):
            if pos > last:
                slices.append(x[:, last:pos, :])
            slices.append(updates[pos].unsqueeze(1))
            last = pos + 1
        if last < S:
            slices.append(x[:, last:S, :])
        return torch.cat(slices, dim=1)


class CompleteLatticeCore(nn.Module):
    def __init__(self, d_model: int, max_seq_len: int):
        super().__init__()
        self.multi_level  = MultiLevelLatticeProcessor(d_model, max_seq_len)
        self.path_weighted = PathWeightedLatticeCore(d_model, max_seq_len)
        self.meta_fusion  = nn.Sequential(
            nn.Linear(d_model * 3, d_model * 2),
            nn.LayerNorm(d_model * 2), nn.GELU(),
            nn.Linear(d_model * 2, d_model)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h_multi = self.multi_level(x)
        h_path  = self.path_weighted(x)
        return self.meta_fusion(torch.cat([x, h_multi, h_path], dim=-1))


# ── Cell 14: Harmonic Horizon Predictor ─────────────────────
class HarmonicHorizonPredictor(nn.Module):
    def __init__(self, d_model: int, vocab_size: int, horizon: int = 8):
        super().__init__()
        self.horizon  = horizon
        d_mid         = d_model // 2
        self.proj_down = nn.Linear(d_model, d_mid)
        self.proj_up   = nn.Linear(d_mid, d_mid * horizon)
        self.norm      = nn.LayerNorm(d_mid)
        self.prediction_head = nn.Linear(d_mid, vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        x_last    = x[:, -1, :]
        down      = F.gelu(self.proj_down(x_last))
        projected = self.proj_up(down).view(-1, self.horizon, down.shape[-1])
        projected = self.norm(projected)
        logits    = self.prediction_head(projected)
        confidence = torch.ones(x_last.shape[0], self.horizon,
                                device=x_last.device, dtype=x_last.dtype)
        return logits, confidence


# ══════════════════════════════════════════════════════════
#  COMBINED MODEL: IOv52-HST
# ══════════════════════════════════════════════════════════

class IOHSTCombined(nn.Module):
    def __init__(self, cfg: type = Config):
        super().__init__()
        self.cfg      = cfg
        self.n_bottom = cfg.N_LAYERS // 2
        n_chunks      = cfg.MAX_SEQ_LEN // cfg.CHUNK_SIZE

        self.token_emb = nn.Embedding(cfg.VOCAB_SIZE, cfg.D_MODEL)
        self.pos_emb   = nn.Embedding(cfg.MAX_SEQ_LEN * 2, cfg.D_MODEL)

        self.bottom = nn.ModuleList([
            AdaptiveBlock(
                cfg.D_MODEL, cfg.N_HEADS,
                use_ssm=(i < cfg.N_MAMBA_LAYERS),
                use_hebbian=(i < cfg.N_MAMBA_LAYERS),
            )
            for i in range(self.n_bottom)
        ])

        self.cif = CIFModule(cfg.D_MODEL, cfg.CIF_THRESHOLD)

        self.chunk_encoder = ChunkEncoder(
            cfg.D_MODEL, chunk_size=cfg.CHUNK_SIZE,
            n_heads=max(1, cfg.N_HEADS // 2),
            n_layers=cfg.N_CHUNK_ENC_LAYERS,
        )

        self.lattice = CompleteLatticeCore(cfg.D_MODEL, n_chunks)

        self.diamond = DiamondCrossAttention(cfg.D_MODEL, cfg.N_HEADS)

        self.top = nn.ModuleList([
            TransformerBlock(cfg.D_MODEL, cfg.N_HEADS,
                             use_diamond_ffn=(i % 2 == 1))
            for i in range(cfg.N_TOP_LAYERS)
        ])

        self.chunk_decoder = ChunkDecoderWithCache(
            cfg.D_MODEL, cfg.VOCAB_SIZE,
            chunk_size=cfg.CHUNK_SIZE,
            n_heads=max(1, cfg.N_HEADS // 2),
            n_layers=cfg.N_CHUNK_DEC_LAYERS,
        )

        self.horizon_predictor = HarmonicHorizonPredictor(
            cfg.D_MODEL, cfg.VOCAB_SIZE, cfg.HORIZON
        )

        self.ln_f    = nn.LayerNorm(cfg.D_MODEL)
        self.lm_head = nn.Linear(cfg.D_MODEL, cfg.VOCAB_SIZE, bias=False)
        self.lm_head.weight = self.token_emb.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        B, S = input_ids.shape
        past_len = past_key_values[0][0].size(2) if past_key_values else 0
        full_S   = S + past_len

        pos_ids = torch.arange(past_len, full_S,
                               dtype=torch.long, device=input_ids.device)
        h = self.token_emb(input_ids) + self.pos_emb(pos_ids)
        new_past: List[Tuple[torch.Tensor, torch.Tensor]] = []

        use_ckpt = self.training and torch.is_grad_enabled()
        def _make_ckpt_fn(b_, p_):
            def _fn(h_):
                out, conf, pres = b_(h_, p_)
                return out, conf, pres[0], pres[1]
            return _fn

        for i, block in enumerate(self.bottom):
            past = past_key_values[i] if past_key_values else None
            if use_ckpt:
                h, _conf, pk, pv = torch.utils.checkpoint.checkpoint(
                    _make_ckpt_fn(block, past), h, use_reentrant=False)
                present = (pk, pv)
            else:
                h, _conf, present = block(h, past)
            new_past.append(present)
        bottom_h = h

        cif_alphas = None
        if S > 1:
            h_cif_raw, cif_alphas = self.cif(h)
            h_cif_up = F.interpolate(
                h_cif_raw.transpose(1, 2),
                size=S, mode="linear", align_corners=False
            ).transpose(1, 2)
            h_enriched = bottom_h + 0.1 * h_cif_up
        else:
            h_enriched = bottom_h

        n_chunks   = max(1, S // self.cfg.CHUNK_SIZE)
        target_len = n_chunks * self.cfg.CHUNK_SIZE
        if S >= target_len:
            enc_input = h_enriched[:, :target_len, :]
        else:
            enc_input = F.pad(h_enriched.transpose(1, 2),
                               (0, target_len - S)).transpose(1, 2)

        chunk_emb = self.chunk_encoder(enc_input)
        h_lattice = self.lattice(chunk_emb)
        h_bridged = self.diamond(h_lattice, bottom_h)
        h_top = h_bridged
        for blk in self.top:
            h_top, _ = blk(h_top)

        logits, _ = self.chunk_decoder(h_top, enc_input)
        if logits.shape[1] > S:
            logits = logits[:, :S, :]
        elif logits.shape[1] < S:
            logits = F.pad(logits, (0, 0, 0, S - logits.shape[1]))

        logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)

        horizon_logits, horizon_conf = self.horizon_predictor(h_top)

        loss         = None
        horizon_loss = None
        if labels is not None:
            shift_l = logits[..., :-1, :].contiguous()
            shift_t = labels[..., 1:].contiguous()
            loss    = F.cross_entropy(
                shift_l.view(-1, self.cfg.VOCAB_SIZE),
                shift_t.view(-1)
            )
            if logits.shape[1] > self.cfg.HORIZON:
                h_targets = labels[:, 1:self.cfg.HORIZON + 1].contiguous()
                h_logits  = horizon_logits[:, :h_targets.shape[1], :]
                horizon_loss = F.cross_entropy(
                    h_logits.reshape(-1, self.cfg.VOCAB_SIZE),
                    h_targets.reshape(-1)
                )

        return {
            "logits":        logits,
            "past_key_values": new_past,
            "loss":          loss,
            "cif_alphas":    cif_alphas,
            "horizon_logits": horizon_logits,
            "horizon_loss":  horizon_loss,
        }


# ══════════════════════════════════════════════════════════
#  DATASET
# ══════════════════════════════════════════════════════════

def build_fineweb_corpus(
    tokenizer: GPT2Tokenizer,
    device: torch.device,
    max_tokens: int = 5_000_000,
    local_test: bool = False,
    stream_timeout: int = 60,
) -> torch.Tensor:
    if local_test:
        max_tokens = 50_000
    CHUNK_CHARS = 200_000
    MIN_LEN     = 200

    all_chunks: List[torch.Tensor] = []
    total_tok   = 0

    def _try_stream(dataset_kwargs: dict) -> bool:
        nonlocal total_tok
        import threading, queue as _queue

        row_q: _queue.Queue = _queue.Queue(maxsize=32)
        stop_event = threading.Event()

        def _producer():
            try:
                ds = load_dataset(**dataset_kwargs)
                for row in ds:
                    if stop_event.is_set():
                        break
                    row_q.put(row)
                row_q.put(None)
            except Exception as e:
                row_q.put(e)

        t = threading.Thread(target=_producer, daemon=True)
        t.start()

        buf = ""
        got_first = False
        last_heartbeat = time.time()

        while True:
            try:
                row = row_q.get(timeout=stream_timeout)
            except _queue.Empty:
                stop_event.set()
                if not got_first:
                    print(f"\n  [timeout] No rows in {stream_timeout}s — switching dataset.", flush=True)
                    return False
                if buf:
                    tok = tokenizer.encode(buf, return_tensors="pt")
                    all_chunks.append(tok); total_tok += tok.shape[1]
                return True

            if row is None:
                if buf:
                    tok = tokenizer.encode(buf, return_tensors="pt")
                    all_chunks.append(tok); total_tok += tok.shape[1]
                return True
            if isinstance(row, Exception):
                stop_event.set()
                print(f"\n  [stream error] {row}", flush=True)
                return False

            got_first = True
            text = row.get("text", "")
            if len(text.strip()) < MIN_LEN:
                continue
            buf += " " + text
            if time.time() - last_heartbeat > 10:
                print(f"  … {total_tok:,} tokens buffered", end="\r", flush=True)
                last_heartbeat = time.time()
            if len(buf) >= CHUNK_CHARS:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    tok = tokenizer.encode(buf, return_tensors="pt")
                all_chunks.append(tok)
                total_tok += tok.shape[1]
                buf = ""
                print(f"  … {total_tok:,} tokens", end="\r", flush=True)
                if total_tok >= max_tokens:
                    stop_event.set()
                    return True
        return True

    print(f"Streaming FineWeb-Edu (target: {max_tokens:,} tokens)...", flush=True)
    success = _try_stream(dict(
        path="HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        streaming=True,
        trust_remote_code=True,
    ))

    if not success or not all_chunks:
        print(f"\nFineWeb-Edu failed — trying WikiText-103...", flush=True)
        success = _try_stream(dict(
            path="wikitext",
            name="wikitext-103-raw-v1",
            split="train",
            streaming=True,
        ))

    if not all_chunks:
        print("Both streams failed — using built-in fallback corpus.", flush=True)
        fallback = [
            "Knowledge distillation trains a student model to mimic a teacher. "
            "The KL divergence between soft probability distributions guides learning. "
            "Hierarchical models compress sequences into chunk-level representations. "
            "State space models represent sequences using continuous dynamic systems. "
            "Pure autoregressive training maximises the log-likelihood of next tokens. "
        ] * 500
        tok = tokenizer.encode(" ".join(fallback), return_tensors="pt")
        all_chunks.append(tok)

    tokens = torch.cat(all_chunks, dim=1).to(device)
    print(f"\nCorpus ready: {tokens.shape[1]:,} tokens on {device}", flush=True)
    return tokens


def get_batch(tokens: torch.Tensor, batch_size: int, seq_len: int) -> torch.Tensor:
    max_start = tokens.shape[1] - seq_len - 1
    if max_start <= 0:
        return tokens[:, :seq_len + 1].expand(batch_size, -1)
    starts = torch.randint(0, max_start, (batch_size,))
    return torch.stack([tokens[0, s: s + seq_len + 1] for s in starts])


# ══════════════════════════════════════════════════════════
#  LR SCHEDULER + LLRD OPTIMIZER
# ══════════════════════════════════════════════════════════

def cosine_lr(step: int, total: int, warmup: int, base: float) -> float:
    if step < warmup:
        return base * (step + 1) / warmup
    p = (step - warmup) / max(1, total - warmup)
    return base * 0.5 * (1.0 + math.cos(math.pi * p))


LLRD_DECAY = 0.85

def build_llrd_groups(model: IOHSTCombined, base_lr: float) -> list:
    groups = []
    for name, mod in [("chunk_encoder",    model.chunk_encoder),
                      ("lattice",           model.lattice),
                      ("diamond",           model.diamond),
                      ("chunk_decoder",     model.chunk_decoder),
                      ("horizon_predictor", model.horizon_predictor),
                      ("ln_f",              model.ln_f)]:
        params = [p for p in mod.parameters() if p.requires_grad]
        if params:
            groups.append({"params": params, "lr": base_lr, "name": name})
    n_top = len(model.top)
    for i, blk in enumerate(model.top):
        factor = LLRD_DECAY ** (n_top - 1 - i)
        params = [p for p in blk.parameters() if p.requires_grad]
        if params:
            groups.append({"params": params, "lr": base_lr * factor, "name": f"top_{i}"})
    cif_params = [p for p in model.cif.parameters() if p.requires_grad]
    if cif_params:
        groups.append({"params": cif_params, "lr": base_lr * 0.7, "name": "cif"})
    n = len(model.bottom)
    for i, blk in enumerate(model.bottom):
        factor = LLRD_DECAY ** (n - 1 - i)
        params = [p for p in blk.parameters() if p.requires_grad]
        if params:
            groups.append({"params": params, "lr": base_lr * factor, "name": f"bottom_{i}"})
    emb = [p for p in list(model.token_emb.parameters()) +
           list(model.pos_emb.parameters()) if p.requires_grad]
    if emb:
        groups.append({"params": emb, "lr": base_lr * 0.3, "name": "embeddings"})
    return groups


# ══════════════════════════════════════════════════════════
#  ENHANCED NAN/INF HANDLING
# ══════════════════════════════════════════════════════════

def sanitize_gradients(model: nn.Module, optimizer: torch.optim.Optimizer) -> bool:
    has_nan = False
    for p in model.parameters():
        if p.grad is not None and not torch.isfinite(p.grad).all():
            has_nan = True
            p.grad.zero_()
    if has_nan:
        for group in optimizer.param_groups:
            for p in group["params"]:
                state = optimizer.state.get(p, {})
                if "exp_avg" in state:
                    state["exp_avg"].zero_()
                    state["exp_avg_sq"].zero_()
    return has_nan


# ══════════════════════════════════════════════════════════
#  HF UPLOAD FUNCTION
# ══════════════════════════════════════════════════════════

def upload_checkpoint_to_hf(model, tokenizer, step, checkpoint_dir="/kaggle/working/checkpoints"):
    """Upload model checkpoint to Hugging Face Hub"""
    if not HF_TOKEN:
        print(f"⚠️  No HF token found, skipping upload at step {step}", flush=True)
        return
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_step_{step}")
    os.makedirs(checkpoint_path, exist_ok=True)
    
    # Save model and tokenizer
    torch.save(model.state_dict(), os.path.join(checkpoint_path, "model.pt"))
    tokenizer.save_pretrained(checkpoint_path)
    
    # Save training info
    info = {
        "step": step,
        "model_config": {
            "d_model": Config.D_MODEL,
            "n_heads": Config.N_HEADS,
            "n_layers": Config.N_LAYERS,
            "n_top_layers": Config.N_TOP_LAYERS,
            "vocab_size": Config.VOCAB_SIZE,
            "max_seq_len": Config.MAX_SEQ_LEN,
        }
    }
    with open(os.path.join(checkpoint_path, "training_info.json"), "w") as f:
        json.dump(info, f, indent=2)
    
    # Upload to HF
    try:
        api = HfApi()
        
        # Upload all files in the checkpoint folder
        for root, dirs, files in os.walk(checkpoint_path):
            for file in files:
                local_path = os.path.join(root, file)
                relative_path = os.path.relpath(local_path, checkpoint_path)
                hf_path = f"checkpoint_step_{step}/{relative_path}"
                
                api.upload_file(
                    path_or_fileobj=local_path,
                    path_in_repo=hf_path,
                    repo_id=Config.HF_REPO_ID,
                    repo_type="model",
                    commit_message=f"Upload checkpoint at step {step}"
                )
        
        print(f"✅ Uploaded checkpoint to {Config.HF_REPO_ID} at step {step}", flush=True)
        
        # Also upload as latest
        for root, dirs, files in os.walk(checkpoint_path):
            for file in files:
                local_path = os.path.join(root, file)
                relative_path = os.path.relpath(local_path, checkpoint_path)
                hf_path = f"latest/{relative_path}"
                
                api.upload_file(
                    path_or_fileobj=local_path,
                    path_in_repo=hf_path,
                    repo_id=Config.HF_REPO_ID,
                    repo_type="model",
                    commit_message=f"Upload latest checkpoint at step {step}"
                )
        
        print(f"✅ Updated 'latest' checkpoint", flush=True)
        
    except Exception as e:
        print(f"❌ Failed to upload checkpoint at step {step}: {e}", flush=True)
    
    # Clean up local checkpoint to save space
    shutil.rmtree(checkpoint_path)


# ══════════════════════════════════════════════════════════
#  TRAINING LOOP
# ══════════════════════════════════════════════════════════

def train(
    model: IOHSTCombined,
    tokenizer: GPT2Tokenizer,
    corpus: torch.Tensor,
    n_steps: int = None,
) -> None:
    cfg        = Config
    steps      = n_steps if n_steps is not None else cfg.TRAIN_STEPS
    fwd_model  = model
    if steps == 0:
        print("steps=0, skipping.", flush=True)
        return

    import gc

    n_bottom = model.n_bottom
    lr_top   = cfg.LR
    lr_bot0  = cfg.LR * 0.1 * (LLRD_DECAY ** (n_bottom - 1))
    lr_emb   = cfg.LR * 0.01

    print(f"\n{'='*65}", flush=True)
    print(f"  IOv52-HST 2B — Pure Training — {steps} steps  [LLRD]", flush=True)
    print(f"  D={cfg.D_MODEL} | H={cfg.N_HEADS} | Bottom={n_bottom}"
          f" | Top={cfg.N_TOP_LAYERS} | ChunkSize={cfg.CHUNK_SIZE}", flush=True)
    print(f"  LLRD={LLRD_DECAY} | lr_top={lr_top:.1e} | lr_bot0={lr_bot0:.1e}"
          f" | lr_emb={lr_emb:.1e}", flush=True)
    print(f"  Warmup={cfg.WARMUP_STEPS} | AMP={cfg.USE_AMP}"
          f" | Batch={cfg.BATCH_SIZE}×{cfg.GRAD_ACCUM}={cfg.BATCH_SIZE*cfg.GRAD_ACCUM} eff"
          f" | SeqLen={cfg.MAX_SEQ_LEN}", flush=True)
    print(f"  Loss: CE + CIF({cfg.CIF_LOSS_WEIGHT}) + Horizon({cfg.HORIZON_LOSS_WEIGHT})", flush=True)
    print(f"  HF Upload: every {cfg.CHECKPOINT_INTERVAL} steps to {cfg.HF_REPO_ID}", flush=True)
    print(f"{'='*65}\n", flush=True)

    fwd_model.train()

    param_groups = build_llrd_groups(model, cfg.LR)

    try:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(
            param_groups, weight_decay=cfg.WEIGHT_DECAY,
            eps=1e-8, betas=(0.9, 0.95),
        )
        print(f"  Optimizer: AdamW 8-bit (bitsandbytes) — saves ~4 GB", flush=True)
    except ImportError:
        optimizer = torch.optim.AdamW(
            param_groups, weight_decay=cfg.WEIGHT_DECAY,
            eps=1e-8, betas=(0.9, 0.95),
        )
        print("  Optimizer: AdamW fp32 (bitsandbytes not available)", flush=True)

    use_scaler = torch.cuda.is_available()
    kw     = {} if _AMP_DEVICE is None else {"device": _AMP_DEVICE}
    scaler = GradScaler(**kw, enabled=use_scaler)
    amp_kw = {} if _AMP_DEVICE is None else {"device_type": _AMP_DEVICE}
    print(f"  Training mode: fp32 + autocast AMP + GradScaler", flush=True)

    base_lrs  = [pg["lr"] for pg in optimizer.param_groups]
    best_loss = float("inf")
    oom_count = 0
    nan_count = 0
    t0        = time.time()

    for step in range(steps):
        scale = cosine_lr(step, steps, cfg.WARMUP_STEPS, 1.0)
        for pg, blr in zip(optimizer.param_groups, base_lrs):
            pg["lr"] = blr * scale

        optimizer.zero_grad(set_to_none=True)
        accum_ce = accum_cif = accum_hor = 0.0
        step_ok  = True

        for _micro in range(cfg.GRAD_ACCUM):
            batch     = get_batch(corpus, cfg.BATCH_SIZE, cfg.MAX_SEQ_LEN)
            input_ids = batch[:, :-1]
            labels    = batch[:, 1:]

            try:
                with autocast(**amp_kw, enabled=use_scaler):
                    out    = fwd_model(input_ids, labels=labels)
                    logits = out["logits"]
                    logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)

                ce_loss = F.cross_entropy(
                    logits.float().reshape(-1, cfg.VOCAB_SIZE),
                    labels.reshape(-1),
                )

                cif_loss = torch.tensor(0.0, device=cfg.DEVICE)
                if out["cif_alphas"] is not None:
                    target_sum = float(cfg.MAX_SEQ_LEN // cfg.CHUNK_SIZE)
                    raw_cif    = ((out["cif_alphas"].float().sum(-1) - target_sum) ** 2).mean()
                    cif_loss   = torch.nan_to_num(raw_cif, nan=0.0, posinf=0.0, neginf=0.0)

                hor_loss = torch.tensor(0.0, device=cfg.DEVICE)
                if out["horizon_loss"] is not None:
                    hl = out["horizon_loss"].float()
                    if torch.isfinite(hl):
                        hor_loss = hl
                    else:
                        print(f"\n⚠️  NaN horizon loss at step {step+1} — zeroing", flush=True)

                hor_ramp = min(1.0, step / max(1, int(steps * 0.6)))

                loss = (
                    ce_loss
                    + cfg.CIF_LOSS_WEIGHT                * cif_loss
                    + cfg.HORIZON_LOSS_WEIGHT * hor_ramp * hor_loss
                ) / cfg.GRAD_ACCUM

                if not torch.isfinite(loss):
                    print(f"\n⚠️  Non-finite loss at step {step+1} micro {_micro}: "
                          f"ce={ce_loss.item():.4f} hor={hor_loss.item():.4f} — skipping microstep.",
                          flush=True)
                    step_ok = False
                    break

                scaler.scale(loss).backward()
                accum_ce  += ce_loss.item()
                accum_cif += cif_loss.item()
                accum_hor += hor_loss.item()

            except torch.cuda.OutOfMemoryError:
                oom_count += 1
                print(f"\n⚠️  OOM at step {step+1} micro {_micro} "
                      f"(total OOMs: {oom_count}) — skipping step.",
                      flush=True)
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache(); gc.collect()
                step_ok = False
                break

            except RuntimeError as e:
                if "nan" in str(e).lower() or "inf" in str(e).lower():
                    print(f"\n⚠️  NaN/Inf in forward/backward at step {step+1}: {str(e)[:100]}", flush=True)
                    step_ok = False
                    break
                raise

            del out, logits, loss, ce_loss, cif_loss, hor_loss, batch, input_ids, labels

        if not step_ok:
            nan_count += 1
            if nan_count > 10:
                print(f"\n❌ Too many NaN steps ({nan_count}) — stopping training.", flush=True)
                break
            continue

        scaler.unscale_(optimizer)

        if sanitize_gradients(model, optimizer):
            nan_count += 1
            print(f"\n⚠️  NaN/inf gradient at step {step+1} — resetting optimizer state and skipping.", flush=True)
            optimizer.zero_grad(set_to_none=True)
            scaler.update()
            torch.cuda.empty_cache()
            if nan_count > 10:
                print(f"\n❌ Too many NaN gradient steps ({nan_count}) — stopping training.", flush=True)
                break
            continue

        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()

        if step % 50 == 0:
            torch.cuda.empty_cache(); gc.collect()

        ce_v   = accum_ce  / cfg.GRAD_ACCUM
        cif_v  = accum_cif / cfg.GRAD_ACCUM
        hor_v  = accum_hor / cfg.GRAD_ACCUM
        hor_ramp_v = min(1.0, step / max(1, int(steps * 0.6)))
        loss_v = ce_v + cfg.CIF_LOSS_WEIGHT * cif_v + cfg.HORIZON_LOSS_WEIGHT * hor_ramp_v * hor_v

        trend = "↓" if loss_v < best_loss else "↑"
        if loss_v < best_loss:
            best_loss = loss_v
            nan_count = max(0, nan_count - 1)

        # Upload checkpoint every 1000 steps
        if (step + 1) % cfg.CHECKPOINT_INTERVAL == 0:
            print(f"\n📤 Uploading checkpoint at step {step+1}...", flush=True)
            upload_checkpoint_to_hf(model, tokenizer, step+1)
            print(f"✅ Checkpoint uploaded\n", flush=True)

        if (step + 1) % 10 == 0 or step < 5:
            elapsed = time.time() - t0
            lr_now  = optimizer.param_groups[0]["lr"]
            mem_gb  = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
            print(
                f"Step {step+1:4d}/{steps} {trend} | "
                f"loss={loss_v:.4f} best={best_loss:.4f} | "
                f"ce={ce_v:.4f} hor={hor_v:.3f}(×{hor_ramp_v:.2f}) | "
                f"lr={lr_now:.2e} | mem={mem_gb:.1f}GB | {elapsed:.0f}s",
                flush=True,
            )
    
    # Final upload
    print(f"\n📤 Uploading final checkpoint...", flush=True)
    upload_checkpoint_to_hf(model, tokenizer, steps)
    print(f"\nTraining complete. Best loss: {best_loss:.4f}", flush=True)


# ── Generation helpers ──────────────────────────────────────
def apply_rep_penalty(logits: torch.Tensor, generated: torch.Tensor, p: float):
    if p == 1.0:
        return logits
    for tid in generated[0].unique():
        if logits[0, tid] > 0:
            logits[0, tid] /= p
        else:
            logits[0, tid] *= p
    return logits


def top_p_sample(logits: torch.Tensor, temp: float, top_p: float):
    probs = F.softmax(logits / temp, dim=-1)
    sp, si = torch.sort(probs, descending=True)
    cum = torch.cumsum(sp, dim=-1)
    mask = cum > top_p
    mask[:, 1:] = mask[:, :-1].clone()
    mask[:, 0]  = False
    probs[0, si[0, mask.squeeze()]] = 0.0
    probs = probs / probs.sum(dim=-1, keepdim=True)
    return torch.multinomial(probs, num_samples=1)


@torch.no_grad()
def generate(model: IOHSTCombined, tokenizer: GPT2Tokenizer) -> str:
    model.eval()
    cfg = Config
    generated = tokenizer.encode(cfg.PROMPT_TEXT, return_tensors="pt").to(cfg.DEVICE)
    input_ids = generated[:, -cfg.MAX_SEQ_LEN:]
    out       = model(input_ids)
    past      = out["past_key_values"]
    next_logits = out["logits"][:, -1:, :]
    next_logits = next_logits[:, -1, :]
    t0 = time.time()
    for i in range(cfg.MAX_GEN_TOKENS):
        next_logits = apply_rep_penalty(next_logits, generated, cfg.REPETITION_PENALTY)
        next_token  = top_p_sample(next_logits, cfg.GEN_TEMPERATURE, cfg.TOP_P)
        generated   = torch.cat([generated, next_token], dim=-1)
        out         = model(next_token, past_key_values=past)
        past        = out["past_key_values"]
        next_logits = out["logits"][:, -1, :]
        if (i + 1) % 100 == 0:
            print(f"  Generated {i+1}/{cfg.MAX_GEN_TOKENS} tokens "
                  f"({(i+1)/(time.time()-t0):.1f} tok/s)", flush=True)
    return tokenizer.decode(generated.squeeze(), skip_special_tokens=True)


# ══════════════════════════════════════════════════════════
#  LOCAL SMOKE TEST
# ══════════════════════════════════════════════════════════

def run_local_test():
    print("\n" + "="*60, flush=True)
    print("  LOCAL SMOKE TEST (5 steps, CPU/small GPU)", flush=True)
    print("="*60, flush=True)

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    orig_steps  = Config.TRAIN_STEPS
    orig_batch  = Config.BATCH_SIZE
    orig_accum  = Config.GRAD_ACCUM
    orig_amp    = Config.USE_AMP
    orig_tokens = Config.MAX_GEN_TOKENS
    orig_checkpoint_interval = Config.CHECKPOINT_INTERVAL

    Config.TRAIN_STEPS    = 5
    Config.BATCH_SIZE     = 1
    Config.GRAD_ACCUM     = 1
    Config.USE_AMP        = torch.cuda.is_available()
    Config.MAX_GEN_TOKENS = 20
    Config.D_MODEL        = 256
    Config.N_HEADS        = 4
    Config.N_LAYERS       = 4
    Config.N_TOP_LAYERS   = 2
    Config.N_CHUNK_ENC_LAYERS = 1
    Config.N_CHUNK_DEC_LAYERS = 1
    Config.N_LATTICE_LAYERS   = 1
    Config.MAX_SEQ_LEN    = 128
    Config.CHUNK_SIZE     = 32
    Config.N_MAMBA_LAYERS = 1
    Config.SSM_D_STATE    = 8
    Config.CHECKPOINT_INTERVAL = 1000  # Won't trigger in test

    print("Building IOHSTCombined (smoke-test dims)...", flush=True)
    model = IOHSTCombined(Config).to(Config.DEVICE)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Parameters: {n_params:.1f}M (small dims for speed test)", flush=True)

    print("Building mini corpus...", flush=True)
    texts = [
        "Hierarchical models compress sequences into chunk-level representations before refining. "
        "The attention mechanism allows each position to attend to all other positions in context. "
        "State space models represent sequences using continuous dynamic systems with hidden states. "
        "Pure autoregressive training maximises the log-likelihood of the next token at each step. "
    ] * 30
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        toks = tokenizer.encode(" ".join(texts), return_tensors="pt").to(Config.DEVICE)
    print(f"Mini corpus: {toks.shape[1]:,} tokens", flush=True)

    train(model, tokenizer, toks, n_steps=5)

    print("\nRunning short generation test...", flush=True)
    text = generate(model, tokenizer)
    print(f"Generated: {text[:200]}...", flush=True)

    Config.TRAIN_STEPS        = orig_steps
    Config.BATCH_SIZE         = orig_batch
    Config.GRAD_ACCUM         = orig_accum
    Config.USE_AMP            = orig_amp
    Config.MAX_GEN_TOKENS     = orig_tokens
    Config.D_MODEL            = 1280
    Config.N_HEADS            = 10
    Config.N_LAYERS           = 24
    Config.N_TOP_LAYERS       = 4
    Config.N_CHUNK_ENC_LAYERS = 2
    Config.N_CHUNK_DEC_LAYERS = 2
    Config.N_LATTICE_LAYERS   = 3
    Config.MAX_SEQ_LEN        = 256
    Config.CHUNK_SIZE         = 32
    Config.N_MAMBA_LAYERS     = 3
    Config.SSM_D_STATE        = 16
    Config.CHECKPOINT_INTERVAL = orig_checkpoint_interval

    print("\n✅ Smoke test PASSED — all shapes valid, training loop works.\n", flush=True)
    return model, tokenizer


# ══════════════════════════════════════════════════════════
#  KAGGLE FULL RUN
# ══════════════════════════════════════════════════════════

def kaggle_full_run():
    import gc

    print("\n" + "="*70, flush=True)
    print("  IOHSTCombined — Pure Training (FIXED NaN HANDLING + HF Auto-upload)", flush=True)
    print("="*70, flush=True)

    n_gpu = torch.cuda.device_count()
    print(f"GPUs available: {n_gpu}", flush=True)

    total_vram = 0.0
    if torch.cuda.is_available():
        for i in range(n_gpu):
            free, total = torch.cuda.mem_get_info(i)
            total_vram += total / 1e9
            print(f"  GPU {i}: {free/1e9:.1f} GB free / {total/1e9:.1f} GB total", flush=True)
        print(f"Total VRAM: {total_vram:.1f} GB", flush=True)

    vram_per_gpu = total_vram / max(n_gpu, 1)
    Config.BATCH_SIZE  = 2 if vram_per_gpu >= 14.0 else 1
    Config.TRAIN_STEPS = 5000  # Set to 5000
    Config.WARMUP_STEPS = 200
    print(f"Auto-config: BATCH_SIZE={Config.BATCH_SIZE} TRAIN_STEPS={Config.TRAIN_STEPS}", flush=True)

    torch.cuda.empty_cache(); gc.collect()

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    print("Building model...", flush=True)
    model = IOHSTCombined(Config).to(Config.DEVICE)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    w_gb   = n_params * 1e6 * 4 / 1e9
    opt_gb = n_params * 1e6 * 2 / 1e9
    print(f"Parameters: {n_params:.1f}M | weights={w_gb:.1f} GB fp32 "
          f"+ 8-bit Adam≈{opt_gb:.1f} GB | est total≈{w_gb+opt_gb+2:.1f} GB", flush=True)

    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1e9
        print(f"GPU 0 after model load: {alloc:.2f} GB allocated", flush=True)

    corpus = build_fineweb_corpus(tokenizer, Config.DEVICE)
    train(model, tokenizer, corpus, n_steps=Config.TRAIN_STEPS)

    print("\nSaving model weights...", flush=True)
    torch.save(model.state_dict(), "/kaggle/working/iohst_weights.pt")
    print("Saved: /kaggle/working/iohst_weights.pt", flush=True)

    print("\nGenerating sample text...", flush=True)
    text = generate(model, tokenizer)
    with open(f"/kaggle/working/{Config.OUTPUT_FILENAME}", "w") as f:
        f.write(text)
    print(f"\n--- First 500 chars ---\n{text[:500]}\n---", flush=True)


def main():
    print("PHASE 1: Local smoke test (5 steps)...", flush=True)
    run_local_test()
    print("\n✅ Local test passed. Ready for Kaggle run.", flush=True)
    print("\nTo run on Kaggle T4 x2, call kaggle_full_run()", flush=True)


KAGGLE_RUN = os.environ.get("KAGGLE_KERNEL_RUN_TYPE") is not None

if __name__ == "__main__" or KAGGLE_RUN:
    if KAGGLE_RUN:
        kaggle_full_run()
    else:
        main()
