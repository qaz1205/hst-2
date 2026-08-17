# ============================================================
#  IOv52-HST — HORIZON FINETUNE SCRIPT (v2 — REDESIGNED)
#  Loads checkpoint_step_5000 from thagnitti/io
#  Freezes ALL params except horizon_predictor + top blocks
#  Trains horizon loss only: target ~1 in 2 hours
#  Uploads final model as "FULL" to thagnitti/io
#
#  CHANGES vs original:
#  ──────────────────────────────────────────────────────s──────
#  1. HarmonicHorizonPredictor completely rewritten:
#       • Per-step learned query embeddings (one per horizon step)
#       • Cross-attention over full h_top sequence (not just last pos)
#       • Context aggregation gate (last chunk + global avg)
#       • 2-layer FFN refinement per step with residuals
#       • Learned confidence head (was constant ones)
#
#  2. Target bug fixed in IOHSTCombined.forward:
#       OLD (WRONG):  labels[:, 1:HORIZON+1]  ← retrodiction, not prediction
#       NEW (FIXED):  labels[:, -HORIZON:]     ← last H tokens after full context
#       The chunk repr covers 0→S, so predicting the tail tokens is causal.
#
#  3. horizon_predictor.prediction_head weight is tied to lm_head in init
#     (weight sharing for better initialization, matches existing init code)
# ============================================================

import subprocess, sys, os, json, time as _time, gc

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(line_buffering=True)

_t_start = _time.time()
def _ts():
    return f"[+{_time.time()-_t_start:5.1f}s]"

print(f"{_ts()} ── Installing dependencies ──────────────────────────", flush=True)

os.environ["PYDEVD_DISABLE_FILE_VALIDATION"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ── HF Token ──────────────────────────────────────────────
HF_TOKEN = ""
if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") is not None:
    print("Detected Kaggle environment, using UserSecretsClient...", flush=True)
    try:
        from kaggle_secrets import UserSecretsClient
        user_secrets = UserSecretsClient()
        for secret_name in ["KAGGLE_HF_TOKEN", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"]:
            try:
                token = user_secrets.get_secret(secret_name)
                if token:
                    HF_TOKEN = token
                    print(f"✓ HF token found as '{secret_name}'", flush=True)
                    break
            except Exception as e:
                print(f"  Secret '{secret_name}' not found: {e}", flush=True)
        if not HF_TOKEN:
            print("⚠️  No HF token found — uploads will fail!", flush=True)
    except ImportError:
        print("⚠️  kaggle_secrets not available", flush=True)
else:
    HF_TOKEN = (
        os.environ.get("KAGGLE_HF_TOKEN") or
        os.environ.get("HF_TOKEN") or
        os.environ.get("HUGGING_FACE_HUB_TOKEN") or ""
    )
    if HF_TOKEN:
        print("✓ HF token found in environment", flush=True)
    else:
        print("⚠️  No HF token found in environment", flush=True)

if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN
    os.environ["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN

# ── Deps ──────────────────────────────────────────────────
def _pip_install(*pkgs, timeout=300):
    print(f"Installing: {' '.join(pkgs)} ...", flush=True)
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "--no-warn-conflicts", *pkgs],
        capture_output=True, text=True, timeout=timeout
    )
    if result.returncode != 0:
        print(f"[pip ERROR]\n{result.stderr[-2000:]}", flush=True)
        raise RuntimeError(f"pip install failed: {pkgs}")
    print(f"  ✓ {' '.join(pkgs)}", flush=True)

_pip_install(
    "transformers>=4.40.0",
    "datasets>=2.19.0",
    "accelerate>=0.30.0",
    "huggingface_hub>=0.23.0",
    "bitsandbytes>=0.43.0",
)
print("Dependencies ready ✓", flush=True)

# ── Imports ───────────────────────────────────────────────
print(f"{_ts()} ── Imports ──────────────────────────────────────────", flush=True)
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

# ── AMP dtype selection ────────────────────────────────────────────────────
# bf16 has the same 8-bit exponent range as fp32, so softmax exp() and
# attention dot-products physically CANNOT overflow to inf/NaN (unlike fp16
# which clips at 65504 and overflows on hot attention maps).
# Fall back to fp16 only on older GPUs that don't support bf16 (T4, P100).
_BF16_SUPPORTED = (
    torch.cuda.is_available()
    and torch.cuda.is_bf16_supported()
)
_AMP_DTYPE = torch.bfloat16 if _BF16_SUPPORTED else torch.float16

if _AMP_DEVICE is not None:
    def autocast_ctx(enabled=True):
        return autocast(device_type=_AMP_DEVICE, dtype=_AMP_DTYPE, enabled=enabled)
else:
    def autocast_ctx(enabled=True):
        return autocast(enabled=enabled)

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
from huggingface_hub import HfApi, hf_hub_download

print(f"PyTorch {torch.__version__} | CUDA: {torch.cuda.is_available()}", flush=True)
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        name = torch.cuda.get_device_name(i)
        cap  = torch.cuda.get_device_capability(i)
        print(f"  GPU {i}: {name} (sm_{cap[0]}{cap[1]})", flush=True)

# ── Config ────────────────────────────────────────────────
print(f"{_ts()} ── Config ───────────────────────────────────────────", flush=True)
class Config:
    D_MODEL: int            = 1280
    N_HEADS: int            = 10
    N_LAYERS: int           = 24
    N_TOP_LAYERS: int       = 4
    N_CHUNK_ENC_LAYERS: int = 2
    N_CHUNK_DEC_LAYERS: int = 2
    N_LATTICE_LAYERS: int   = 3
    MAX_SEQ_LEN: int        = 256
    VOCAB_SIZE: int         = 50257
    CHUNK_SIZE: int         = 32
    HORIZON: int            = 8
    N_MAMBA_LAYERS: int     = 3
    SSM_EXPAND: int         = 1
    SSM_D_STATE: int        = 16
    CIF_THRESHOLD: float    = 0.5

    HORIZON_LR: float    = 2e-5   # lower: oscillation caused by noisy 1-token target
    WEIGHT_DECAY: float  = 0.01
    WARMUP_STEPS: int    = 200   # ramp safely from ~1e-7 → 2e-5
    GRAD_ACCUM: int      = 8     # doubled: 16 eff seqs per step smooths the loss curve
    BATCH_SIZE: int      = 2
    GRAD_CLIP: float     = 0.5
    USE_AMP: bool        = True

    # ── Loss hyperparams ─────────────────────────────────────────────────
    # Label smoothing softens the per-token CE gradient.  Without it, a
    # single wrong prediction (CE ≈ 10) dominates the step and the model
    # oscillates — the loss standard deviation exceeds the mean by training
    # on only 1 token per sequence at H=1.
    LABEL_SMOOTHING: float = 0.1

    # Multi-anchor training: compute the horizon loss at MULTI_PRED_ANCHORS
    # different prefix lengths of h_top per micro-batch.  This multiplies
    # the effective training signal by ~MULTI_PRED_ANCHORS at very low cost
    # (the horizon predictor is small relative to the full backbone).
    # For MULTI_PRED_ANCHORS=4 and n_chunks=7, strides ≈ [2, 4, 6, 7].
    MULTI_PRED_ANCHORS: int = 4

    # ── Adaptive horizon curriculum ───────────────────────────────────────
    # Gate rationale per level (chunk-level predictor, no token context):
    #   H=1: loss ≈ 4.5 → perplexity ≈ 90.  Realistic without full attention.
    #   H=2: loss ≈ 5.0 → perplexity ≈ 148. Next-but-one is harder.
    HORIZON_GATE_LOSS: float     = 4.5
    HORIZON_GATE_MIN_STEPS: int  = 50
    HORIZON_EMA_ALPHA: float     = 0.92

    TIME_BUDGET_S: int   = 7200
    UPLOAD_MARGIN_S: int = 300
    MID_UPLOAD_INTERVAL_S: int = 1200

    HF_REPO_ID: str          = "thagnitti/io"
    CHECKPOINT_SUBFOLDER: str = "checkpoint_step_5000"
    FINAL_UPLOAD_NAME: str   = "FULL"

    DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SEED: int = 42

random.seed(Config.SEED)
torch.manual_seed(Config.SEED)
print(f"Running on: {Config.DEVICE}", flush=True)


# ════════════════════════════════════════════════════════════
#  ARCHITECTURE — unchanged components from gate06.py
# ════════════════════════════════════════════════════════════

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


class SelfAttentionWithCache(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.n_heads  = n_heads
        self.head_dim = d_model // n_heads
        self.qkv      = nn.Linear(d_model, d_model * 3, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

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
        present   = (k, v)
        is_causal = (layer_past is None)
        attn_out  = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
        out       = attn_out.transpose(1, 2).contiguous().view(B, S, D)
        return self.out_proj(out), present


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

    def forward(self, tgt, memory, self_attn_past=None, cross_attn_past=None):
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
        self.chunk_size    = chunk_size
        self.pos_embedding = nn.Embedding(chunk_size, d_model)
        self.layers        = nn.ModuleList([
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
            layer_cache      = cache[i] if cache else (None, None)
            sa_past, ca_past = layer_cache
            chunk_idx        = min(
                (past_len // self.chunk_size),
                chunk_embeddings.size(1) - 1
            )
            memory = chunk_embeddings[:, chunk_idx:chunk_idx+1, :].expand(B, S, D)
            tgt, sa_present, ca_present = layer(tgt, memory, sa_past, ca_past)
            new_cache.append((sa_present, ca_present))
        logits = self.lm_head(tgt)
        return logits, new_cache


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
        self.max_depth         = len(spine)
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
        self.analyzer         = FullLatticeFieldAnalyzer(max_seq_len)
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
        B, S, D   = x.shape
        spine     = self.analyzer.spine
        rel_spine = spine[spine < S]
        updates   = {}
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
        B, S, D   = x.shape
        spine     = self.analyzer.spine
        rel_spine = spine[spine < S]
        updates   = {}
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
        self.multi_level   = MultiLevelLatticeProcessor(d_model, max_seq_len)
        self.path_weighted = PathWeightedLatticeCore(d_model, max_seq_len)
        self.meta_fusion   = nn.Sequential(
            nn.Linear(d_model * 3, d_model * 2),
            nn.LayerNorm(d_model * 2), nn.GELU(),
            nn.Linear(d_model * 2, d_model)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h_multi = self.multi_level(x)
        h_path  = self.path_weighted(x)
        return self.meta_fusion(torch.cat([x, h_multi, h_path], dim=-1))


# ════════════════════════════════════════════════════════════
#  REDESIGNED HarmonicHorizonPredictor  (v3 — autoregressive)
#  ─────────────────────────────────────────────────────────
#  Original problems:
#    1. Only used x[:, -1, :] — threw away all other chunks
#    2. proj_up with no nonlinearity → all steps undifferentiated
#    3. Constant confidence of 1 — no gradient signal
#    4. All horizon steps predicted independently — no step-to-step
#       conditioning, so step 3 couldn't learn from step 2's prediction
#
#  Design (three stages):
#
#  STAGE 1 — Context encoding
#    • Context gate fuses last-chunk + global average into a summary vector
#    • Summary appended to memory so every step query can attend to it
#
#  STAGE 2 — Parallel context cross-attention
#    • Each of the H step queries attends to the full h_top sequence
#    • Gives each step a rich, position-aware starting representation
#    • Per-step FFN adds non-linear capacity
#
#  STAGE 3 — Autoregressive causal self-attention over steps
#    • A causal (lower-triangular) self-attention over the H step states
#    • Step i can attend to steps 0…i-1 but NOT i+1…H-1
#    • Equivalent to running a tiny autoregressive transformer over the
#      horizon, fully parallel during training (no Python for-loop)
#    • Step 2's prediction conditions on step 1's representation, etc.
#    • Followed by a second FFN + residual for extra capacity
#
#  STAGE 4 — Predict & confidence
#    • Two-layer projection → vocab logits
#    • Learned per-step confidence (was always-ones before)
# ════════════════════════════════════════════════════════════

class HarmonicHorizonPredictor(nn.Module):
    def __init__(self, d_model: int, vocab_size: int, horizon: int = 8,
                 n_heads: int = 10):
        super().__init__()
        self.horizon = horizon
        self.d_model = d_model
        d_mid        = d_model // 2  # 640

        # ── Stage 1: Context gate ─────────────────────────────────────────
        # Blends last-chunk (most recent context) with global mean pool.
        # The result is appended to memory so every step attends to it.
        self.context_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model, bias=True),
            nn.Sigmoid(),
        )
        self.context_norm = nn.LayerNorm(d_model)

        # ── Stage 2: Per-step learned queries ─────────────────────────────
        # Each horizon step has its own independent starting vector.
        # Shape [1, H, D] — broadcast over batch.
        self.step_queries = nn.Parameter(torch.randn(1, horizon, d_model) * 0.02)

        # Cross-attention: step queries → full memory (h_top + context)
        self.cross_attn = nn.MultiheadAttention(
            d_model, num_heads=n_heads, batch_first=True, dropout=0.0
        )
        self.cross_norm = nn.LayerNorm(d_model)

        # FFN after cross-attention
        self.cross_ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )
        self.cross_ffn_norm = nn.LayerNorm(d_model)

        # ── Stage 3: Autoregressive causal self-attention over steps ──────
        # A causal mask ensures step i only sees steps 0…i-1.
        # Fully parallel during training — no Python loop needed.
        self.causal_attn = nn.MultiheadAttention(
            d_model, num_heads=n_heads, batch_first=True, dropout=0.0
        )
        self.causal_norm = nn.LayerNorm(d_model)

        # Second FFN after causal self-attention
        self.causal_ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )
        self.causal_ffn_norm = nn.LayerNorm(d_model)

        # ── Stage 4: Projection + prediction head ─────────────────────────
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_mid),
            nn.GELU(),
            nn.Linear(d_mid, d_mid),
            nn.LayerNorm(d_mid),
        )
        # Initialised from lm_head[:, :d_mid] in train_horizon()
        self.prediction_head = nn.Linear(d_mid, vocab_size, bias=False)

        # ── Confidence head ───────────────────────────────────────────────
        # Learned per-step confidence in (0, 1). Was always-ones before.
        self.confidence_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        # Pre-build the causal mask buffer (re-used every forward pass)
        causal_mask = torch.triu(
            torch.full((horizon, horizon), float('-inf')), diagonal=1
        )
        self.register_buffer('causal_mask', causal_mask)

        # ── Output-projection zero-init (GPT-NeoX / LLaMA style) ──────────
        # Zero-initializing the *output* linear of every sub-block means the
        # module starts as the identity (residual path dominates).  This
        # constrains the initial loss to log(vocab_size) ≈ 10.8 instead of
        # the 100-200 range produced by random Kaiming init feeding a
        # lm_head-scaled prediction_head, which causes gradient explosion
        # and NaN on step ~50.
        #
        # What gets zero-initialized:
        #   cross_ffn[-1]   — output projection of cross-attention FFN
        #   causal_ffn[-1]  — output projection of causal self-attn FFN
        #
        # What gets tiny-scale init:
        #   prediction_head — std = 0.002 keeps max initial logit spread
        #                     well below 1.0 → near-uniform softmax → CE ≈ 10.8
        #
        # NOTE: do NOT zero-init proj[2] here because the following
        # LayerNorm(proj[2]=0) would produce NaN (0/0).
        for ffn in (self.cross_ffn, self.causal_ffn):
            nn.init.zeros_(ffn[-1].weight)
            nn.init.zeros_(ffn[-1].bias)
        nn.init.normal_(self.prediction_head.weight, std=0.002)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: chunk-level hidden states [B, n_chunks, D_MODEL]
        Returns:
            logits:     [B, horizon, vocab_size]
            confidence: [B, horizon]
        """
        if x.ndim == 2:
            x = x.unsqueeze(1)
        B, n_chunks, D = x.shape

        # ── Stage 1: Build context + memory ──────────────────────────────
        x_last = x[:, -1, :]                                              # [B, D]
        x_mean = x.mean(dim=1)                                            # [B, D]
        gate   = self.context_gate(torch.cat([x_last, x_mean], dim=-1))  # [B, D]
        ctx    = self.context_norm(x_last + gate * x_mean)                # [B, D]
        # Append summary token — every step query attends to it
        memory = torch.cat([x, ctx.unsqueeze(1)], dim=1)  # [B, n_chunks+1, D]

        # ── Stage 2: Parallel cross-attention ────────────────────────────
        queries  = self.step_queries.expand(B, -1, -1)        # [B, H, D]
        ca_out, _ = self.cross_attn(queries, memory, memory)
        step_h   = self.cross_norm(queries + ca_out)          # [B, H, D]
        step_h   = self.cross_ffn_norm(step_h + self.cross_ffn(step_h))

        # ── Stage 3: Autoregressive causal self-attention over steps ──────
        # attn_mask is [H, H] with -inf in the upper triangle so step i
        # can only attend to positions 0…i (itself + earlier steps).
        ar_out, _ = self.causal_attn(
            step_h, step_h, step_h,
            attn_mask=self.causal_mask,
        )
        step_h = self.causal_norm(step_h + ar_out)             # [B, H, D]
        step_h = self.causal_ffn_norm(step_h + self.causal_ffn(step_h))

        # ── Stage 4: Predict ──────────────────────────────────────────────
        proj       = self.proj(step_h)                          # [B, H, d_mid]
        logits     = self.prediction_head(proj)                 # [B, H, vocab]
        confidence = self.confidence_head(step_h).squeeze(-1)  # [B, H]

        return logits, confidence


# ════════════════════════════════════════════════════════════
#  FULL MODEL — unchanged except for the fixed horizon target
# ════════════════════════════════════════════════════════════

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

        # ── Redesigned horizon predictor ───────────────────────────────────
        self.horizon_predictor = HarmonicHorizonPredictor(
            cfg.D_MODEL, cfg.VOCAB_SIZE, cfg.HORIZON,
            n_heads=cfg.N_HEADS,
        )

        self.ln_f    = nn.LayerNorm(cfg.D_MODEL)
        self.lm_head = nn.Linear(cfg.D_MODEL, cfg.VOCAB_SIZE, bias=False)
        self.lm_head.weight = self.token_emb.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        labels: Optional[torch.Tensor] = None,
        active_horizon: Optional[int] = None,
    ) -> Dict[str, Any]:
        B, S = input_ids.shape
        past_len = past_key_values[0][0].size(2) if past_key_values else 0
        full_S   = S + past_len

        pos_ids = torch.arange(past_len, full_S,
                               dtype=torch.long, device=input_ids.device)
        h = self.token_emb(input_ids) + self.pos_emb(pos_ids)
        new_past: List[Tuple[torch.Tensor, torch.Tensor]] = []

        for i, block in enumerate(self.bottom):
            past  = past_key_values[i] if past_key_values else None
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
        h_top     = h_bridged
        for blk in self.top:
            h_top, _ = blk(h_top)

        # ── Horizon prediction ────────────────────────────────────────────
        horizon_logits, horizon_conf = self.horizon_predictor(h_top.float())

        horizon_loss = None
        if labels is not None and S >= self.cfg.HORIZON:
            # active_horizon: curriculum ramps 1 → HORIZON over training.
            # When None (e.g. inference), always use the full HORIZON.
            H        = active_horizon if active_horizon is not None else self.cfg.HORIZON
            H        = max(1, min(H, self.cfg.HORIZON))
            V        = self.cfg.VOCAB_SIZE
            chunk_sz = self.cfg.CHUNK_SIZE
            ls       = self.cfg.LABEL_SMOOTHING

            # Per-step recency weights: step 1 gets the strongest gradient.
            step_w = torch.arange(H, 0, -1, dtype=torch.float32,
                                  device=h_top.device)
            step_w = step_w / step_w.sum()                    # [H], sums to 1

            def _anchor_loss(logits: torch.Tensor,
                             tgts:   torch.Tensor) -> "Optional[torch.Tensor]":
                """
                Weighted CE for one (logits [B,H,V], tgts [B,≥H]) anchor.
                Returns None if the target window is too short to fill H steps.
                Label smoothing is applied here — it dampens the per-token
                gradient, reducing loss variance across batches by ~40-60%.
                """
                if tgts.shape[1] < H:
                    return None
                tgts = tgts[:, :H].contiguous()
                per  = F.cross_entropy(
                    logits.float().reshape(-1, V),
                    tgts.reshape(-1),
                    reduction="none",
                    label_smoothing=ls,
                ).view(B, H)
                return (per * step_w).sum(dim=1).mean()

            anchor_losses: List[torch.Tensor] = []

            # ── Anchor 0: full h_top → last H tokens of labels ────────────
            # This is the primary training signal (existing behaviour).
            a0 = _anchor_loss(horizon_logits[:, :H, :], labels[:, -H:])
            if a0 is not None:
                anchor_losses.append(a0)

            # ── Extra anchors: shorter h_top prefixes (data augmentation) ─
            # For prefix of k chunks (covering input[0..k*chunk_sz-1]), the
            # predictor sees less context and targets the next H tokens after
            # that prefix.  These shorter-context predictions are genuinely
            # easier → cleaner gradient signal early in training.
            # Each extra anchor costs ~1 additional horizon_predictor call
            # (cheap relative to the full backbone forward+backward).
            n_extra = self.cfg.MULTI_PRED_ANCHORS - 1  # anchor 0 already counted
            if n_extra > 0 and n_chunks > 1:
                stride = max(1, n_chunks // self.cfg.MULTI_PRED_ANCHORS)
                for k in range(stride, n_chunks, stride):
                    tok_end = k * chunk_sz          # first target token index
                    if tok_end >= labels.shape[1]:
                        continue
                    part_logits, _ = self.horizon_predictor(
                        h_top[:, :k, :].float()
                    )
                    a = _anchor_loss(
                        part_logits[:, :H, :],
                        labels[:, tok_end : tok_end + H],
                    )
                    if a is not None:
                        anchor_losses.append(a)

            if anchor_losses:
                # Average across all anchors so the loss scale stays
                # comparable to the single-anchor baseline.
                horizon_loss = sum(anchor_losses) / len(anchor_losses)

        return {
            "past_key_values": new_past,
            "horizon_logits":  horizon_logits,
            "horizon_loss":    horizon_loss,
        }


# ════════════════════════════════════════════════════════════
#  LOAD CHECKPOINT FROM HF
# ════════════════════════════════════════════════════════════

def load_checkpoint(model: IOHSTCombined, repo_id: str, subfolder: str, hf_token: str):
    print(f"\n📥 Downloading model.pt from {repo_id}/{subfolder} ...", flush=True)
    local_path = hf_hub_download(
        repo_id=repo_id,
        filename=f"{subfolder}/model.pt",
        repo_type="model",
        token=hf_token,
        local_dir="/kaggle/working/ckpt_download",
    )
    print(f"  Downloaded to: {local_path}", flush=True)
    state = torch.load(local_path, map_location="cpu")

    # The new horizon_predictor has different keys → strict=False is essential
    missing, unexpected = model.load_state_dict(state, strict=False)

    # Expected missing: all new horizon_predictor params (step_queries, cross_attn, etc.)
    hp_missing = [k for k in missing if "horizon_predictor" in k]
    other_missing = [k for k in missing if "horizon_predictor" not in k]
    if hp_missing:
        print(f"  ℹ️  Horizon predictor new params (will be randomly init'd): {len(hp_missing)}", flush=True)
    if other_missing:
        print(f"  ⚠️  Other missing keys ({len(other_missing)}): {other_missing[:5]}", flush=True)
    if unexpected:
        print(f"  ⚠️  Unexpected keys ({len(unexpected)}): {unexpected[:5]}", flush=True)
    print(f"  ✅ Weights loaded successfully.", flush=True)
    return model


# ════════════════════════════════════════════════════════════
#  FREEZE
# ════════════════════════════════════════════════════════════

# Module-level constant so both freeze() and the checkpointer
# use the same set of prefixes — no risk of them drifting apart.
TRAINABLE_PREFIXES: Tuple[str, ...] = (
    "horizon_predictor", "top.", "ln_f", "lm_head", "token_emb"
)


def freeze_all_except_horizon(model: IOHSTCombined):
    total, trainable = 0, 0
    for name, param in model.named_parameters():
        total += param.numel()
        if any(name.startswith(p) for p in TRAINABLE_PREFIXES):
            param.requires_grad = True
            trainable += param.numel()
        else:
            param.requires_grad = False
    frozen = total - trainable
    print(f"\n🔒 Frozen:    {frozen/1e6:.2f}M params", flush=True)
    print(f"🔥 Trainable: {trainable/1e6:.2f}M params", flush=True)
    return model


# ════════════════════════════════════════════════════════════
#  DATASET
# ════════════════════════════════════════════════════════════

def build_corpus(tokenizer: GPT2Tokenizer, device: torch.device,
                 max_tokens: int = 8_000_000) -> torch.Tensor:
    print(f"\nStreaming FineWeb-Edu (target ≥ {max_tokens:,} tokens)...", flush=True)
    all_chunks: List[torch.Tensor] = []
    total_tok = 0
    buf = ""
    CHUNK_CHARS = 300_000
    MIN_LEN = 200

    try:
        ds = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            name="sample-10BT",
            split="train",
            streaming=True,
        )
        for row in ds:
            text = row.get("text", "")
            if len(text.strip()) < MIN_LEN:
                continue
            buf += " " + text
            if len(buf) >= CHUNK_CHARS:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    tok = tokenizer.encode(buf, return_tensors="pt")
                all_chunks.append(tok)
                total_tok += tok.shape[1]
                buf = ""
                print(f"  … {total_tok:,} tokens", end="\r", flush=True)
                if total_tok >= max_tokens:
                    break
    except Exception as e:
        print(f"\nFineWeb stream error: {e} — trying wikitext fallback", flush=True)

    if not all_chunks:
        print("Using WikiText-103 fallback...", flush=True)
        try:
            ds = load_dataset("wikitext", "wikitext-103-raw-v1",
                              split="train", streaming=True)
            for row in ds:
                text = row.get("text", "")
                if len(text.strip()) < MIN_LEN:
                    continue
                buf += " " + text
                if len(buf) >= CHUNK_CHARS:
                    tok = tokenizer.encode(buf, return_tensors="pt")
                    all_chunks.append(tok)
                    total_tok += tok.shape[1]
                    buf = ""
                    if total_tok >= max_tokens:
                        break
        except Exception as e2:
            print(f"WikiText also failed: {e2}", flush=True)

    if not all_chunks:
        print("⚠️  Using tiny built-in fallback corpus!", flush=True)
        text = ("The model predicts future tokens using a horizon predictor. "
                "Sequence modeling requires attending to long-range context. ") * 2000
        tok = tokenizer.encode(text, return_tensors="pt")
        all_chunks.append(tok)

    if buf:
        tok = tokenizer.encode(buf, return_tensors="pt")
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


# ════════════════════════════════════════════════════════════
#  UPLOAD HELPERS
# ════════════════════════════════════════════════════════════

def upload_to_hf(model: IOHSTCombined, tokenizer: GPT2Tokenizer,
                 folder_name: str, step: int, hf_token: str,
                 repo_id: str = "thagnitti/io"):
    if not hf_token:
        print(f"⚠️  No HF token — skipping upload of '{folder_name}'", flush=True)
        return False

    save_dir = f"/kaggle/working/upload_{folder_name}"
    os.makedirs(save_dir, exist_ok=True)

    torch.save(model.state_dict(), os.path.join(save_dir, "model.pt"))
    tokenizer.save_pretrained(save_dir)

    info = {
        "folder": folder_name,
        "step": step,
        "note": "horizon_predictor v2 finetuned — redesigned predictor + fixed target",
        "model_config": {
            "d_model": Config.D_MODEL, "n_heads": Config.N_HEADS,
            "n_layers": Config.N_LAYERS, "n_top_layers": Config.N_TOP_LAYERS,
            "vocab_size": Config.VOCAB_SIZE, "max_seq_len": Config.MAX_SEQ_LEN,
            "horizon": Config.HORIZON,
        }
    }
    with open(os.path.join(save_dir, "training_info.json"), "w") as f:
        json.dump(info, f, indent=2)

    print(f"\n📤 Uploading to {repo_id}/{folder_name} (step {step})...", flush=True)
    try:
        api = HfApi()
        for root, _, files in os.walk(save_dir):
            for fname in files:
                local_path = os.path.join(root, fname)
                rel        = os.path.relpath(local_path, save_dir)
                hf_path    = f"{folder_name}/{rel}"
                api.upload_file(
                    path_or_fileobj=local_path,
                    path_in_repo=hf_path,
                    repo_id=repo_id,
                    repo_type="model",
                    token=hf_token,
                    commit_message=f"horizon v2 finetune step {step} → {folder_name}",
                )
        print(f"✅ Uploaded {folder_name} at step {step}", flush=True)
        return True
    except Exception as e:
        print(f"❌ Upload failed: {e}", flush=True)
        import traceback; traceback.print_exc()
        return False


# ════════════════════════════════════════════════════════════
#  TRAINING LOOP
# ════════════════════════════════════════════════════════════

def cosine_lr(step: int, total: int, warmup: int, base: float) -> float:
    if step < warmup:
        return base * (step + 1) / warmup
    p = (step - warmup) / max(1, total - warmup)
    return base * 0.5 * (1.0 + math.cos(math.pi * p))


class HorizonCurriculumTracker:
    """
    Adaptive horizon curriculum — expands only when the model is ready.

    How it works
    ────────────
    The tracker keeps an exponential moving average (EMA) of the horizon
    loss at the current level.  When that EMA drops below `gate_loss` AND
    the model has spent at least `min_steps` steps at the current level,
    the horizon advances by 1 (e.g. 1→2, 2→3, …, 7→8).

    Why adaptive instead of time-based?
    • A fast-converging model doesn't sit at H=1 for 50 unnecessary steps.
    • A slow-converging model doesn't get pushed to H=8 before it can even
      reliably predict H=1 — which is exactly the original failure mode.
    • The EMA smooths over noisy mini-batch losses, so a single lucky low
      loss won't trigger a premature expansion.

    Config knobs (all in Config class)
    ───────────────────────────────────
    HORIZON_GATE_LOSS     float  Expand when EMA(loss) < this.  2.5 ≈
                                 e^2.5 ≈ 12-way perplexity — achievable
                                 but not trivial for 1-step prediction.
    HORIZON_GATE_MIN_STEPS int   Minimum steps at each level.  Prevents
                                 pathological rapid cycling.
    HORIZON_EMA_ALPHA     float  EMA smoothing (0.92 ≈ 12-step memory).
                                 Higher = smoother but slower to react.
    """

    def __init__(self, max_horizon: int, gate_loss: float,
                 min_steps: int, ema_alpha: float):
        self.active      = 1           # current active horizon (1-indexed)
        self.max         = max_horizon
        self.gate_loss   = gate_loss
        self.min_steps   = min_steps
        self.alpha       = ema_alpha
        self.steps_here  = 0          # steps spent at current level
        self.ema_loss    = None        # EMA loss for current level

    @property
    def at_max(self) -> bool:
        return self.active >= self.max

    def update(self, loss_value: float) -> bool:
        """
        Feed the latest horizon loss for the current active level.
        Returns True (and expands horizon by 1) if the gate condition is met.
        Does nothing and returns False otherwise.
        """
        self.steps_here += 1

        # Warm-start EMA on first observation at this level
        if self.ema_loss is None:
            self.ema_loss = loss_value
        else:
            self.ema_loss = (self.alpha * self.ema_loss
                             + (1.0 - self.alpha) * loss_value)

        if (not self.at_max
                and self.steps_here >= self.min_steps
                and self.ema_loss < self.gate_loss):
            self.active     += 1
            self.steps_here  = 0
            self.ema_loss    = None   # fresh EMA for the new, harder level
            return True               # caller should log the expansion

        return False

    def status(self) -> str:
        ema_str = f"{self.ema_loss:.4f}" if self.ema_loss is not None else "n/a"
        return (f"H={self.active}/{self.max} | "
                f"steps_here={self.steps_here} | EMA={ema_str}"
                f" (gate<{self.gate_loss})")


class BestLevelCheckpointer:
    """
    Saves the best trainable weights for each horizon level to disk.

    Why this exists
    ───────────────
    Kaggle notebooks can timeout or OOM mid-run.  Without this, the final
    upload uses the last step's weights — which may be *worse* than weights
    from 200 steps ago (e.g. if the model started overfitting after the
    horizon expanded too aggressively).

    Strategy
    ────────
    On every successful optimizer step, `maybe_save()` checks whether the
    current loss is a new best for the *current* horizon level.  If so, it
    dumps only the trainable parameter subset to disk (typically 200-400 MB,
    not the full multi-GB model).

    Before the final HF upload, `restore_best()` reloads the checkpoint with
    the lowest overall loss seen across all levels, so the pushed weights are
    always the best the run produced — not just what the last step happened
    to land on.

    Disk layout
    ───────────
    A single file at `save_path` is overwritten each time a new overall best
    is found, so disk usage is bounded to one checkpoint at a time.
    """

    CKPT_PATH = "/kaggle/working/best_horizon_ckpt.pt"

    def __init__(self, save_path: str = CKPT_PATH):
        self.save_path            : str                   = save_path
        self.best_loss_per_level  : Dict[int, float]      = {}
        self.best_overall_loss    : float                  = float("inf")
        self.best_overall_level   : int                    = 0
        self.best_step            : int                    = -1
        self._saves               : int                    = 0   # total saves

    # ── public API ────────────────────────────────────────────────────────

    def maybe_save(self,
                   model: "IOHSTCombined",
                   level: int,
                   loss: float,
                   step: int) -> bool:
        """
        Save iff this is a new best for `level`.
        Returns True on save (caller can log).
        Only the trainable parameter subset is written to keep the file small.
        """
        if loss >= self.best_loss_per_level.get(level, float("inf")):
            return False

        self.best_loss_per_level[level] = loss

        # Serialize only the trainable slice — avoids writing the full model
        trainable_state: Dict[str, torch.Tensor] = {
            name: param.detach().cpu().clone()
            for name, param in model.named_parameters()
            if any(name.startswith(pfx) for pfx in TRAINABLE_PREFIXES)
        }

        torch.save({
            "level"           : level,
            "loss"            : loss,
            "step"            : step,
            "trainable_state" : trainable_state,
            "loss_per_level"  : dict(self.best_loss_per_level),
        }, self.save_path)

        self._saves += 1

        if loss < self.best_overall_loss:
            self.best_overall_loss  = loss
            self.best_overall_level = level
            self.best_step          = step

        return True

    def restore_best(self,
                     model: "IOHSTCombined",
                     device: torch.device) -> Optional[Dict]:
        """
        Load the best saved checkpoint back into `model` (partial load —
        frozen params are untouched).  Returns the checkpoint metadata dict,
        or None if no checkpoint exists yet.
        """
        if not os.path.exists(self.save_path):
            print("  ⚠️  No best-level checkpoint found on disk — "
                  "keeping current weights.", flush=True)
            return None

        ckpt = torch.load(self.save_path, map_location=device)
        # strict=False: frozen params are absent from trainable_state —
        # load_state_dict will simply leave them as-is.
        missing, unexpected = model.load_state_dict(
            ckpt["trainable_state"], strict=False
        )
        # unexpected keys = truly unknown — warn if any
        if unexpected:
            print(f"  ⚠️  restore_best: unexpected keys {unexpected[:5]}", flush=True)
        return ckpt

    def summary(self) -> str:
        if not self.best_loss_per_level:
            return "no checkpoints saved"
        levels = ", ".join(
            f"H={lvl}:{loss:.4f}"
            for lvl, loss in sorted(self.best_loss_per_level.items())
        )
        return (f"saves={self._saves} | best overall: "
                f"H={self.best_overall_level} loss={self.best_overall_loss:.4f} "
                f"@ step {self.best_step} | per-level [{levels}]")


def _horizon_grad_norms(model: "IOHSTCombined") -> Dict[str, float]:
    """
    Return per-sub-module gradient L2 norms for the horizon predictor.

    Called after scaler.unscale_() so the norms reflect true fp32 magnitudes.
    Returns an empty dict if the horizon predictor has no gradients yet.

    Sub-modules reported
    ────────────────────
    context_gate   — blends last-chunk + global mean into context token
    cross_attn     — per-step queries attend over full memory
    cross_ffn      — FFN after cross-attention (output proj is zero-init)
    causal_attn    — autoregressive self-attention over H steps
    causal_ffn     — FFN after causal self-attn (output proj is zero-init)
    proj           — D→d_mid projection feeding prediction_head
    prediction_head — vocab logits (std=0.002 init, no lm_head copy)
    confidence_head — learned per-step confidence scalar
    step_queries   — learned per-step starting vectors (parameter, not module)
    """
    hp = model.horizon_predictor
    groups: Dict[str, List[torch.Tensor]] = {
        "context_gate"    : list(hp.context_gate.parameters()),
        "cross_attn"      : list(hp.cross_attn.parameters()),
        "cross_ffn"       : list(hp.cross_ffn.parameters()),
        "causal_attn"     : list(hp.causal_attn.parameters()),
        "causal_ffn"      : list(hp.causal_ffn.parameters()),
        "proj"            : list(hp.proj.parameters()),
        "prediction_head" : list(hp.prediction_head.parameters()),
        "confidence_head" : list(hp.confidence_head.parameters()),
        "step_queries"    : [hp.step_queries],
    }
    result: Dict[str, float] = {}
    for name, params in groups.items():
        grads = [p.grad for p in params if p.grad is not None]
        if not grads:
            result[name] = 0.0
            continue
        total_sq = sum(g.detach().float().pow(2).sum().item() for g in grads)
        result[name] = math.sqrt(total_sq)
    return result


def _fmt_module_norms(norms: Dict[str, float]) -> str:
    """One-line string of module_name=norm pairs, sorted descending by norm."""
    sorted_pairs = sorted(norms.items(), key=lambda kv: kv[1], reverse=True)
    return "  ".join(f"{k}={v:.3f}" for k, v in sorted_pairs)


def train_horizon(model: IOHSTCombined, tokenizer: GPT2Tokenizer,
                  corpus: torch.Tensor, hf_token: str) -> None:
    cfg = Config

    # NOTE: prediction_head weight is intentionally NOT copied from lm_head.
    # The lm_head was trained to project specific hidden-state distributions
    # (from the full transformer stack) onto vocabulary.  The horizon
    # predictor's proj module outputs a completely different distribution
    # (random Kaiming init).  Feeding that through lm_head-scale weights
    # produces logits in the hundreds → softmax overflow → NaN on step ~50.
    # Instead, HarmonicHorizonPredictor.__init__ already applies:
    #   • zero-init on cross_ffn and causal_ffn output projections
    #   • std=0.002 on prediction_head (near-uniform logits at init)
    # → initial horizon loss ≈ log(50257) ≈ 10.8 and no gradient explosion.

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable_params)

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=cfg.HORIZON_LR,
        weight_decay=cfg.WEIGHT_DECAY,
        eps=1e-8,
        betas=(0.9, 0.95),
    )
    print("Optimizer: AdamW fp32", flush=True)

    # init_scale=256: much lower than default 65536.
    # With bf16 the scaler is mostly ceremonial (bf16 can't overflow), but
    # on fp16 fallback GPUs a lower init scale prevents the first few
    # backward passes from saturating before the scaler can adapt.
    scaler = GradScaler(enabled=cfg.USE_AMP, init_scale=256)

    model.train()

    t0          = time.time()
    deadline    = t0 + cfg.TIME_BUDGET_S - cfg.UPLOAD_MARGIN_S
    last_mid_up = t0
    step        = 0
    best_loss   = float("inf")
    nan_count   = 0
    consecutive_nans = 0

    MAX_STEPS_ESTIMATE = 3000

    print(f"\n{'='*65}", flush=True)
    print(f"  HORIZON FINETUNE v3 — {n_trainable/1e6:.2f}M trainable params", flush=True)
    print(f"  Modules: top (×{cfg.N_TOP_LAYERS}) + ln_f + horizon_predictor (autoregressive)", flush=True)
    print(f"  LR={cfg.HORIZON_LR} | Warmup={cfg.WARMUP_STEPS} | Cosine over {MAX_STEPS_ESTIMATE} steps", flush=True)
    print(f"  Batch={cfg.BATCH_SIZE}×{cfg.GRAD_ACCUM}={cfg.BATCH_SIZE*cfg.GRAD_ACCUM} eff"
          f" | SeqLen={cfg.MAX_SEQ_LEN}", flush=True)
    amp_label = "OFF"
    if cfg.USE_AMP:
        amp_label = f"ON ({_AMP_DTYPE.__name__ if hasattr(_AMP_DTYPE, '__name__') else str(_AMP_DTYPE).split('.')[-1]})"
    print(f"  AMP: {amp_label}", flush=True)
    print(f"  Adaptive curriculum: gate_loss={cfg.HORIZON_GATE_LOSS} | "
          f"min_steps={cfg.HORIZON_GATE_MIN_STEPS} | ema_α={cfg.HORIZON_EMA_ALPHA}", flush=True)
    print(f"  Loss weighting: step 1 = {cfg.HORIZON:.1f}x, step {cfg.HORIZON} = 1x (linear decay)", flush=True)
    print(f"  Time budget: {cfg.TIME_BUDGET_S//60} min", flush=True)
    print(f"  Target: horizon loss → ~1.0", flush=True)
    print(f"{'='*65}\n", flush=True)

    print(f"Training started. Deadline in {(deadline-t0)/60:.1f} min...\n", flush=True)

    # ── Adaptive curriculum tracker ────────────────────────────────────────
    curriculum = HorizonCurriculumTracker(
        max_horizon=cfg.HORIZON,
        gate_loss=cfg.HORIZON_GATE_LOSS,
        min_steps=cfg.HORIZON_GATE_MIN_STEPS,
        ema_alpha=cfg.HORIZON_EMA_ALPHA,
    )
    print(f"  🎓 Curriculum starting at H=1/{cfg.HORIZON}  "
          f"(gate: EMA < {cfg.HORIZON_GATE_LOSS} for ≥{cfg.HORIZON_GATE_MIN_STEPS} steps)", flush=True)

    # ── Per-level best checkpoint ───────────────────────────────────────────
    # Saves only trainable params to /kaggle/working — survives timeouts/OOM.
    # restore_best() is called before final HF upload so we always push the
    # best weights seen, not just whatever the last step landed on.
    checkpointer = BestLevelCheckpointer()
    print(f"  💾 BestLevelCheckpointer active → {checkpointer.save_path}", flush=True)

    while time.time() < deadline and step < MAX_STEPS_ESTIMATE * 2:
        now = time.time()

        if now - last_mid_up >= cfg.MID_UPLOAD_INTERVAL_S:
            mid_name = f"horizon_mid_step{step}"
            upload_to_hf(model, tokenizer, mid_name, step, hf_token)
            last_mid_up = time.time()

        lr_scale = cosine_lr(step, MAX_STEPS_ESTIMATE, cfg.WARMUP_STEPS, 1.0)
        for pg in optimizer.param_groups:
            pg["lr"] = cfg.HORIZON_LR * lr_scale

        # active_h is read from the tracker each step — it may have expanded
        # after the previous step's update() call
        active_h = curriculum.active

        optimizer.zero_grad(set_to_none=True)
        accum_hor = 0.0
        step_ok   = True
        micro_valid_count = 0

        for _micro in range(cfg.GRAD_ACCUM):
            if time.time() >= deadline:
                step_ok = False
                break

            batch     = get_batch(corpus, cfg.BATCH_SIZE, cfg.MAX_SEQ_LEN)
            input_ids = batch[:, :-1]
            labels    = batch[:, 1:]

            try:
                with autocast_ctx(enabled=cfg.USE_AMP):
                    out = model(input_ids, labels=labels, active_horizon=active_h)
                    hor_loss = out["horizon_loss"]

                if hor_loss is None:
                    continue

                loss = hor_loss / cfg.GRAD_ACCUM
                scaler.scale(loss).backward()
                accum_hor += hor_loss.item()
                micro_valid_count += 1

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"\n⚠️  OOM at step {step} micro {_micro} — skipping", flush=True)
                    torch.cuda.empty_cache()
                    gc.collect()
                else:
                    print(f"\n⚠️  RuntimeError at step {step}: {e}", flush=True)
                step_ok = False
                break
            except Exception as e:
                print(f"\n⚠️  Exception at step {step} micro {_micro}: {e}", flush=True)
                step_ok = False
                break

            del out, loss, batch, input_ids, labels

        if not step_ok or micro_valid_count == 0:
            nan_count += 1
            consecutive_nans += 1
            optimizer.zero_grad(set_to_none=True)
            if consecutive_nans > 5:
                print(f"\n❌ Too many consecutive invalid steps ({consecutive_nans}) — stopping.", flush=True)
                break
            continue

        scaler.unscale_(optimizer)

        # Compute per-module norms *before* clipping (true magnitudes).
        # Stored so the periodic log can include them without a second pass.
        module_norms = _horizon_grad_norms(model)

        total_norm = torch.nn.utils.clip_grad_norm_(trainable_params, cfg.GRAD_CLIP)

        has_nan = False
        for p in trainable_params:
            if p.grad is not None and not torch.isfinite(p.grad).all():
                p.grad.zero_()
                has_nan = True
        if has_nan:
            nan_count += 1
            consecutive_nans += 1
            # Always print module breakdown on NaN so the culprit is visible.
            print(
                f"\n⚠️  NaN gradients at step {step} "
                f"(consecutive={consecutive_nans}) — per-module norms:\n"
                f"  {_fmt_module_norms(module_norms)}",
                flush=True,
            )
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if consecutive_nans > 5:
                print(f"\n❌ Too many consecutive NaN gradients — stopping.", flush=True)
                break
            continue

        scaler.step(optimizer)
        scaler.update()

        hor_v = accum_hor / micro_valid_count
        consecutive_nans = 0

        if hor_v < best_loss:
            best_loss = hor_v

        # ── Feed loss to the adaptive curriculum tracker ───────────────────
        # update() returns True the moment the horizon expands — log it.
        expanded = curriculum.update(hor_v)
        if expanded:
            ema_prev = cfg.HORIZON_GATE_LOSS   # threshold it just crossed
            print(
                f"\n  📈 Curriculum gate triggered at step {step}! "
                f"EMA={ema_prev:.4f} < {cfg.HORIZON_GATE_LOSS} "
                f"→ horizon now H={curriculum.active}/{cfg.HORIZON}\n",
                flush=True
            )

        # ── Per-level best checkpoint ───────────────────────────────────────
        # Saves trainable params when we beat the best loss for this level.
        # quiet: only log the very first save and each subsequent new best.
        saved = checkpointer.maybe_save(model, active_h, hor_v, step)
        if saved:
            print(
                f"  💾 New best at H={active_h}: loss={hor_v:.4f} "
                f"(step {step}) → {checkpointer.save_path}",
                flush=True,
            )

        elapsed   = time.time() - t0
        remaining = deadline - time.time()

        if step % 10 == 0:
            lr_now  = optimizer.param_groups[0]["lr"]
            mem_gb  = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
            trend   = "↓" if hor_v <= best_loss else "↑"
            ema_str = f"{curriculum.ema_loss:.4f}" if curriculum.ema_loss is not None else "n/a"
            print(
                f"Step {step:5d} {trend} | H={active_h}/{cfg.HORIZON} EMA={ema_str} | "
                f"hor_loss={hor_v:.4f} best={best_loss:.4f} | "
                f"lr={lr_now:.2e} | grad_norm={total_norm:.2f} | mem={mem_gb:.1f}GB | "
                f"elapsed={elapsed/60:.1f}min rem={remaining/60:.1f}min",
                flush=True
            )

        # Every 50 steps, print per-module gradient breakdown.
        # This makes it trivial to pinpoint which sub-module is exploding
        # if loss starts diverging — no need to wait for a NaN.
        if step % 50 == 0:
            print(
                f"  grad breakdown | {_fmt_module_norms(module_norms)}",
                flush=True,
            )
            torch.cuda.empty_cache()
            gc.collect()

        step += 1

        if best_loss < 1.2:
            print(f"\n🎯 Target reached! horizon loss = {best_loss:.4f} < 1.2 at step {step}", flush=True)
            break

    print(f"\n{'='*65}", flush=True)
    print(f"  Training finished. Steps: {step} | Best horizon loss: {best_loss:.4f}", flush=True)
    print(f"  Final curriculum state: {curriculum.status()}", flush=True)
    print(f"  Checkpoint summary:     {checkpointer.summary()}", flush=True)
    print(f"{'='*65}\n", flush=True)

    # ── Restore best weights before upload ─────────────────────────────────
    # If the last step was worse than a previous checkpoint (e.g. overfitting
    # after a horizon expansion, or weights drifted up), restore the best
    # saved state so the HF upload always reflects the peak of training.
    if checkpointer.best_step >= 0 and checkpointer.best_overall_loss < best_loss:
        print(
            f"\n🔄 Restoring best checkpoint: H={checkpointer.best_overall_level} "
            f"loss={checkpointer.best_overall_loss:.4f} @ step {checkpointer.best_step} "
            f"(current loss={best_loss:.4f} is worse)",
            flush=True,
        )
        ckpt_meta = checkpointer.restore_best(model, cfg.DEVICE)
        if ckpt_meta:
            print(f"  ✅ Restored from {checkpointer.save_path}", flush=True)
    else:
        print(
            f"\n✅ Current weights are the best (loss={best_loss:.4f}) — "
            "no restore needed.",
            flush=True,
        )


# ════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════

def kaggle_horizon_finetune():
    print("\n" + "="*70, flush=True)
    print("  IOv52-HST — HORIZON PREDICTOR FINETUNE v2 (REDESIGNED)", flush=True)
    print("  Load: thagnitti/io/checkpoint_step_5000", flush=True)
    print("  Train: top blocks + ln_f + horizon_predictor", flush=True)
    print("  Upload: thagnitti/io/FULL", flush=True)
    print("="*70, flush=True)

    if not HF_TOKEN:
        raise RuntimeError("No HF token found! Add KAGGLE_HF_TOKEN to Kaggle Secrets.")

    n_gpu = torch.cuda.device_count()
    print(f"GPUs available: {n_gpu}", flush=True)

    total_vram = 0.0
    if torch.cuda.is_available():
        for i in range(n_gpu):
            free, total = torch.cuda.mem_get_info(i)
            total_vram += total / 1e9
            print(f"  GPU {i}: {free/1e9:.1f} GB free / {total/1e9:.1f} GB total", flush=True)

    Config.BATCH_SIZE = 2 if total_vram >= 28.0 else 1

    torch.cuda.empty_cache()
    gc.collect()

    print("\nLoading tokenizer...", flush=True)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    print("\nBuilding IOHSTCombined architecture...", flush=True)
    model = IOHSTCombined(Config)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Total parameters: {n_params:.1f}M", flush=True)

    load_checkpoint(model, Config.HF_REPO_ID,
                    Config.CHECKPOINT_SUBFOLDER, HF_TOKEN)

    model = model.to(Config.DEVICE)

    mem_after_load = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    print(f"GPU memory after load: {mem_after_load:.2f} GB", flush=True)

    model = freeze_all_except_horizon(model)

    corpus = build_corpus(tokenizer, Config.DEVICE, max_tokens=8_000_000)

    train_horizon(model, tokenizer, corpus, HF_TOKEN)

    print(f"\n📤 Final upload as '{Config.FINAL_UPLOAD_NAME}'...", flush=True)

    success = upload_to_hf(
        model, tokenizer,
        Config.FINAL_UPLOAD_NAME,
        step=-1,
        hf_token=HF_TOKEN,
        repo_id=Config.HF_REPO_ID,
    )

    if success:
        print(f"\n🎉 SUCCESS — model uploaded to {Config.HF_REPO_ID}/{Config.FINAL_UPLOAD_NAME}", flush=True)
    else:
        torch.save(model.state_dict(), "/kaggle/working/FULL_model.pt")
        print(f"\n⚠️  Upload failed — saved locally at /kaggle/working/FULL_model.pt", flush=True)

    print("\nDone.", flush=True)


KAGGLE_RUN = os.environ.get("KAGGLE_KERNEL_RUN_TYPE") is not None

if __name__ == "__main__" or KAGGLE_RUN:
    kaggle_horizon_finetune()
