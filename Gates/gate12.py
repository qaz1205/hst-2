# ============================================================
#  IOv52-HST — HORIZON FINETUNE SCRIPT (v10 — PLATEAU FIX)
#  28-hour budget | 2x Tesla T4 | DataParallel
#
#  FIXES (cumulative):
#   1-4  – force-advance, gate, min_steps, warmup (now 200)
#   5    – removed broken bias-correction warmup
#   6    – HORIZON_LR = 3.0e-5
#   7    – β2 = 0.95, ε = 1e-4
#   8    – warmup = 200
#   9    – logit clamping REMOVED (was causing gradient traps)
#   10   – proj layers initialised with std=0.001
#   11   – diagnostic max-logit print on spike
#   [REPAIR BLACK EGG]: hor_tgts offset, LR continuous plateau, GradScaler native recovery
#   [REPAIR OMEGA ARCH]: Tied lm_head, MULTI_PRED_ANCHORS indexing, Causal Diamond mask
#   [CHECKPOINT FIX]: Dynamic shape filtering for mismatched architecture upgrades
#   [REPAIR DEADLOCK & GRADIENT TRAP]: torch.clamp deleted, dictionary deadlock fixed, threads added
#   [MAX LOSS CEILING]: Raised _MAX_VALID_LOSS to 200.0 to allow un-clamped init gradients to flow
#   [REPAIR OOM SHIELD v2]: Bypassed DataParallel dict-gather leaks, nuked orphaned HuggingFace streaming threads, added aggressive Linux malloc_trim.
#   [v10 PLATEAU FIX]: LR_PLATEAU_END raised to 600 (was 200 = same as WARMUP, giving zero flat region).
#                      Half-LR at warmup boundary to absorb first full-LR step spike.
#                      consecutive_nans threshold raised to 20 for both empty-step and NaN-grad checks.
# ============================================================

import subprocess, sys, os, json, gc, threading, ctypes
import time as _time

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

# ── HF Token ─────────────────────────────────────────────────
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

# ── Deps ──────────────────────────────────────────────────────
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

# ── Imports ───────────────────────────────────────────────────
import time, math, random, warnings, collections
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

# ── Config ────────────────────────────────────────────────────
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

    # ── LR schedule ────────────────────────────
    HORIZON_LR: float       = 3.0e-5
    WEIGHT_DECAY: float     = 0.01
    PROJ_WEIGHT_DECAY: float = 0.01

    WARMUP_STEPS: int       = 200
    LR_PLATEAU_END: int     = 600   # v10 FIX: was 200 (= WARMUP_STEPS), giving zero flat region
    LR_CYCLE_STEPS: int     = 500
    LR_CYCLE_MIN_FRAC: float = 0.10

    GRAD_ACCUM: int         = 16
    BATCH_SIZE: int         = 2   # per GPU
    GRAD_CLIP: float        = 2.0

    # ── Curriculum ────────────────────────────────────────────
    HORIZON_GATE_LOSS: float      = 7.8
    HORIZON_GATE_MIN_STEPS: int   = 100
    HORIZON_FORCE_ADVANCE_STEPS: int = 9999  # disabled
    HORIZON_EMA_ALPHA: float      = 0.92

    USE_AMP: bool             = True
    LABEL_SMOOTHING: float    = 0.05
    MULTI_PRED_ANCHORS: int   = 4
    GRADIENT_NOISE_STD: float = 0.0

    TIME_BUDGET_S: int       = 28 * 3600
    UPLOAD_MARGIN_S: int     = 600
    MID_UPLOAD_INTERVAL_S: int = 3600

    HF_REPO_ID: str          = "thagnitti/io"
    CHECKPOINT_SUBFOLDER: str = "checkpoint_step_5000"
    FINAL_UPLOAD_NAME: str   = "FULL"

    DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SEED: int = 42

random.seed(Config.SEED)
torch.manual_seed(Config.SEED)
print(f"Running on: {Config.DEVICE}", flush=True)


# ══════════════════════════════════════════════════════════════
#  ARCHITECTURE
# ══════════════════════════════════════════════════════════════

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
            self.ffn = DiamondMixer(d_model)
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
        B, S_top, D = top_h.shape
        S_bot = bottom_h.size(1)
        q = self.q(top_h).view(B, S_top, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k(bottom_h).view(B, S_bot, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v(bottom_h).view(B, S_bot, self.n_heads, self.head_dim).transpose(1, 2)

        chunk_sz = max(1, S_bot // S_top)
        idx_top = torch.arange(S_top, device=top_h.device).unsqueeze(1)
        idx_bot = torch.arange(S_bot, device=top_h.device).unsqueeze(0)

        mask = idx_bot < (idx_top + 1) * chunk_sz
        if S_bot % S_top != 0:
            mask[-1, :] = True

        attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask.unsqueeze(0).unsqueeze(0))
        out = attn_out.transpose(1, 2).contiguous().view(B, S_top, D)
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

    def get_structure(self, pos: int):
        if pos in self.lattice_structure:
            return self.lattice_structure[pos]
        return self._analyze_non_spine(pos)

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


class HarmonicHorizonPredictor(nn.Module):
    def __init__(self, d_model: int, vocab_size: int, horizon: int = 8,
                 n_heads: int = 10):
        super().__init__()
        self.horizon = horizon
        self.d_model = d_model

        self.context_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model, bias=True),
            nn.Sigmoid(),
        )
        self.context_norm = nn.LayerNorm(d_model)
        self.step_queries = nn.Parameter(torch.randn(1, horizon, d_model) * 0.02)
        self.cross_attn = nn.MultiheadAttention(
            d_model, num_heads=n_heads, batch_first=True, dropout=0.0
        )
        self.cross_norm = nn.LayerNorm(d_model)
        self.cross_ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )
        self.cross_ffn_norm = nn.LayerNorm(d_model)
        self.causal_attn = nn.MultiheadAttention(
            d_model, num_heads=n_heads, batch_first=True, dropout=0.0
        )
        self.causal_norm = nn.LayerNorm(d_model)
        self.causal_ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )
        self.causal_ffn_norm = nn.LayerNorm(d_model)

        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(),
            nn.Linear(d_model, d_model), nn.LayerNorm(d_model),
        )
        for layer in self.proj:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, std=0.001)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

        self.confidence_head = nn.Sequential(
            nn.Linear(d_model, 64), nn.GELU(),
            nn.Linear(64, 1), nn.Sigmoid(),
        )
        causal_mask = torch.triu(
            torch.full((horizon, horizon), float('-inf')), diagonal=1
        )
        self.register_buffer('causal_mask', causal_mask)
        for ffn in (self.cross_ffn, self.causal_ffn):
            if isinstance(ffn[-1], nn.Linear):
                nn.init.zeros_(ffn[-1].weight)
                if ffn[-1].bias is not None:
                    nn.init.zeros_(ffn[-1].bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        B, n_chunks, D = x.shape
        x_last = x[:, -1, :]
        x_mean = x.mean(dim=1)
        gate   = self.context_gate(torch.cat([x_last, x_mean], dim=-1))
        ctx    = self.context_norm(x_last + gate * x_mean)
        memory = torch.cat([x, ctx.unsqueeze(1)], dim=1)
        queries  = self.step_queries.expand(B, -1, -1)
        ca_out, _ = self.cross_attn(queries, memory, memory)
        step_h   = self.cross_norm(queries + ca_out)
        step_h   = self.cross_ffn_norm(step_h + self.cross_ffn(step_h))
        ar_out, _ = self.causal_attn(
            step_h, step_h, step_h, attn_mask=self.causal_mask,
        )
        step_h = self.causal_norm(step_h + ar_out)
        step_h = self.causal_ffn_norm(step_h + self.causal_ffn(step_h))

        proj       = self.proj(step_h)
        confidence = self.confidence_head(step_h).squeeze(-1)
        return proj, confidence


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
        labels: Optional[torch.Tensor] = None,
        horizon_targets: Optional[torch.Tensor] = None,
        active_horizon: Optional[int] = None,
        past_key_values=None,
    ) -> Any:
        B, S = input_ids.shape
        past_len = past_key_values[0][0].size(2) if past_key_values else 0
        full_S   = S + past_len
        pos_ids  = torch.arange(past_len, full_S, dtype=torch.long,
                                device=input_ids.device)
        h = self.token_emb(input_ids) + self.pos_emb(pos_ids)
        new_past = []
        for i, block in enumerate(self.bottom):
            past  = past_key_values[i] if past_key_values else None
            h, _conf, present = block(h, past)
            new_past.append(present)
        bottom_h = h
        if S > 1:
            h_cif_raw, _ = self.cif(h)
            h_cif_up = F.interpolate(
                h_cif_raw.transpose(1, 2), size=S,
                mode="linear", align_corners=False
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

        horizon_features, horizon_conf = self.horizon_predictor(h_top.float())
        horizon_logits = self.lm_head(self.ln_f(horizon_features))

        horizon_loss = None
        if horizon_targets is not None and S >= self.cfg.CHUNK_SIZE:
            H        = active_horizon if active_horizon is not None else self.cfg.HORIZON
            H        = max(1, min(H, self.cfg.HORIZON))
            V        = self.cfg.VOCAB_SIZE
            chunk_sz = self.cfg.CHUNK_SIZE
            ls       = self.cfg.LABEL_SMOOTHING
            step_w = torch.arange(H, 0, -1, dtype=torch.float32,
                                  device=h_top.device)
            step_w = step_w / step_w.sum()

            def _anchor_loss(logits, tgts):
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
            anchor_weights: List[float] = []

            a0 = _anchor_loss(horizon_logits[:, :H, :], horizon_targets[:, :H])
            if a0 is not None:
                anchor_losses.append(a0); anchor_weights.append(1.0)

            n_extra = self.cfg.MULTI_PRED_ANCHORS - 1
            if n_extra > 0 and n_chunks > 1 and labels is not None:
                full_targets = torch.cat([labels, horizon_targets], dim=1)
                stride = max(1, n_chunks // self.cfg.MULTI_PRED_ANCHORS)

                for k in range(stride, n_chunks, stride):
                    tok_end = k * chunk_sz
                    if tok_end >= S:
                        continue

                    tgts_k = full_targets[:, tok_end - 1 : tok_end - 1 + H]

                    part_features, _ = self.horizon_predictor(h_top[:, :k, :].float())
                    part_logits = self.lm_head(self.ln_f(part_features))

                    a = _anchor_loss(part_logits[:, :H, :], tgts_k)
                    if a is not None:
                        anchor_losses.append(a); anchor_weights.append(0.5)

            if anchor_losses:
                total_w = sum(anchor_weights)
                horizon_loss = sum(
                    w * l for w, l in zip(anchor_weights, anchor_losses)
                ) / total_w

        # Return RAW TUPLE during training to prevent DataParallel dictionary CPU leak
        if labels is not None:
            if horizon_loss is None:
                horizon_loss = torch.tensor(0.0, device=h_top.device, requires_grad=True)
            logit_max = horizon_logits.detach().abs().max() if horizon_logits is not None else torch.tensor(0.0, device=h_top.device)
            return horizon_loss, logit_max

        return {
            "past_key_values": new_past,
            "horizon_logits":  horizon_logits,
            "horizon_conf":    horizon_conf,
            "horizon_loss":    horizon_loss,
        }


# ══════════════════════════════════════════════════════════════
#  LOAD / FREEZE / DATASET / UPLOAD
# ══════════════════════════════════════════════════════════════

def load_checkpoint(model: IOHSTCombined, repo_id: str, subfolder: str, hf_token: str):
    print(f"\n📥 Downloading model.pt from {repo_id}/{subfolder} ...", flush=True)
    local_path = hf_hub_download(
        repo_id=repo_id, filename=f"{subfolder}/model.pt",
        repo_type="model", token=hf_token,
        local_dir="/kaggle/working/ckpt_download",
    )
    state = torch.load(local_path, map_location="cpu")

    current_state = model.state_dict()
    filtered_state = {}
    for k, v in state.items():
        if k in current_state and current_state[k].shape != v.shape:
            print(f"  ⚠️  Dropping {k} due to shape mismatch: ckpt {v.shape} vs model {current_state[k].shape}", flush=True)
            continue
        filtered_state[k] = v

    missing, unexpected = model.load_state_dict(filtered_state, strict=False)
    hp_missing    = [k for k in missing if "horizon_predictor" in k]
    other_missing = [k for k in missing if "horizon_predictor" not in k]

    if hp_missing:
        print(f"  ℹ️  Horizon predictor new params: {len(hp_missing)}", flush=True)
    if other_missing:
        print(f"  ⚠️  Other missing: {other_missing[:5]}", flush=True)
    print("  ✅ Weights loaded.", flush=True)
    return model


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
    print(f"\n🔒 Frozen:    {frozen/1e6:.2f}M", flush=True)
    print(f"🔥 Trainable: {trainable/1e6:.2f}M", flush=True)
    return model


def build_corpus(tokenizer, device, max_tokens=8_000_000):
    print(f"\nStreaming FineWeb-Edu (target ≥ {max_tokens:,} tokens)...", flush=True)
    all_chunks, total_tok, buf = [], 0, ""
    CHUNK_CHARS, MIN_LEN = 300_000, 200
    try:
        ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
        ds_iter = iter(ds)
        for row in ds_iter:
            text = row.get("text", "")
            if len(text.strip()) < MIN_LEN:
                continue
            buf += " " + text
            if len(buf) >= CHUNK_CHARS:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    tok = tokenizer.encode(buf, return_tensors="pt")
                all_chunks.append(tok); total_tok += tok.shape[1]; buf = ""
                print(f"  … {total_tok:,} tokens", end="\r", flush=True)
                if total_tok >= max_tokens:
                    break
        del ds_iter
        del ds
    except Exception as e:
        print(f"\nFineWeb error: {e} — wikitext fallback", flush=True)
        try:
            ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train", streaming=True)
            ds_iter = iter(ds)
            for row in ds_iter:
                text = row.get("text", "")
                if len(text.strip()) < MIN_LEN:
                    continue
                buf += " " + text
                if len(buf) >= CHUNK_CHARS:
                    tok = tokenizer.encode(buf, return_tensors="pt")
                    all_chunks.append(tok); total_tok += tok.shape[1]; buf = ""
                    if total_tok >= max_tokens:
                        break
            del ds_iter
            del ds
        except Exception as e2:
            print(f"WikiText also failed: {e2}", flush=True)

    if not all_chunks:
        text = ("The model predicts future tokens using a horizon predictor. "
                "Sequence modeling requires long-range context. ") * 3000
        all_chunks.append(tokenizer.encode(text, return_tensors="pt"))
    if buf:
        all_chunks.append(tokenizer.encode(buf, return_tensors="pt"))

    tokens = torch.cat(all_chunks, dim=1).to(device)
    del all_chunks, buf

    gc.collect()
    try: ctypes.CDLL("libc.so.6").malloc_trim(0)
    except: pass

    print(f"\nCorpus ready: {tokens.shape[1]:,} tokens on {device}", flush=True)
    return tokens


def get_batch(tokens, batch_size, seq_len, horizon):
    row_len   = seq_len + horizon + 1
    max_start = tokens.shape[1] - row_len
    if max_start <= 0:
        row = tokens[0, :row_len]
        if row.shape[0] < row_len:
            row = row.repeat((row_len // row.shape[0]) + 1)[:row_len]
        return row.unsqueeze(0).expand(batch_size, -1)
    starts = torch.randint(0, max_start, (batch_size,))
    return torch.stack([tokens[0, s: s + row_len] for s in starts])


def _bg_upload(save_dir, folder_name, step, hf_token, repo_id):
    try:
        api = HfApi()
        for root, _, files in os.walk(save_dir):
            for fname in files:
                lp  = os.path.join(root, fname)
                rel = os.path.relpath(lp, save_dir)
                api.upload_file(
                    path_or_fileobj=lp,
                    path_in_repo=f"{folder_name}/{rel}",
                    repo_id=repo_id, repo_type="model", token=hf_token,
                    commit_message=f"v10 step {step} → {folder_name}",
                )
        print(f"\n✅ Background upload complete: {folder_name} at step {step}", flush=True)
    except Exception as e:
        print(f"\n❌ Background upload failed: {e}", flush=True)


def upload_to_hf(raw_model, tokenizer, folder_name, step, hf_token, repo_id="thagnitti/io", sync=False):
    if not hf_token:
        print(f"⚠️  No HF token — skipping '{folder_name}'", flush=True)
        return False
    save_dir = f"/kaggle/working/upload_{folder_name}"
    os.makedirs(save_dir, exist_ok=True)
    torch.save(raw_model.state_dict(), os.path.join(save_dir, "model.pt"))
    tokenizer.save_pretrained(save_dir)
    with open(os.path.join(save_dir, "training_info.json"), "w") as f:
        json.dump({"folder": folder_name, "step": step,
                   "note": "v10: plateau fix, half-LR at warmup boundary"}, f, indent=2)

    print(f"\n📤 Spawning upload to {repo_id}/{folder_name} (step {step})...", flush=True)
    if sync:
        _bg_upload(save_dir, folder_name, step, hf_token, repo_id)
    else:
        t = threading.Thread(target=_bg_upload, args=(save_dir, folder_name, step, hf_token, repo_id))
        t.daemon = True
        t.start()
    return True


# ══════════════════════════════════════════════════════════════
#  LR SCHEDULE
# ══════════════════════════════════════════════════════════════

def cyclic_cosine_lr(step, warmup, plateau_end, cycle_steps, base, min_frac=0.10):
    if step < warmup:
        return base * (step + 1) / warmup
    if step < plateau_end:
        return base
    pos = (step - plateau_end) % cycle_steps
    p   = pos / cycle_steps
    return base * (min_frac + (1.0 - min_frac) * 0.5 * (1.0 + math.cos(math.pi * p)))


# ══════════════════════════════════════════════════════════════
#  CURRICULUM & CHECKPOINTING
# ══════════════════════════════════════════════════════════════

class HorizonCurriculumTracker:
    def __init__(self, max_horizon, gate_loss, min_steps,
                 ema_alpha, force_advance_steps):
        self.active              = 1
        self.max                 = max_horizon
        self.gate_loss           = gate_loss
        self.min_steps           = min_steps
        self.force_advance_steps = force_advance_steps
        self.alpha               = ema_alpha
        self.steps_here          = 0
        self.ema_loss            = None
        self.best_ema_here       = float("inf")
        self.steps_no_improve    = 0

    @property
    def at_max(self):
        return self.active >= self.max

    def update(self, loss_value: float) -> Tuple[bool, str]:
        self.steps_here += 1
        if self.ema_loss is None:
            self.ema_loss = loss_value
        else:
            self.ema_loss = self.alpha * self.ema_loss + (1 - self.alpha) * loss_value

        if self.ema_loss < self.best_ema_here:
            self.best_ema_here    = self.ema_loss
            self.steps_no_improve = 0
        else:
            self.steps_no_improve += 1

        if self.at_max:
            return False, ""

        if (self.steps_here >= self.min_steps
                and self.ema_loss < self.gate_loss):
            self.active += 1
            self.steps_here = 0
            self.ema_loss = None
            self.best_ema_here = float("inf")
            self.steps_no_improve = 0
            return True, "gate"

        if self.steps_no_improve >= self.force_advance_steps:
            self.active += 1
            self.steps_here = 0
            self.ema_loss = None
            self.best_ema_here = float("inf")
            self.steps_no_improve = 0
            return True, "force"

        return False, ""

    def status(self):
        ema_str = f"{self.ema_loss:.4f}" if self.ema_loss is not None else "n/a"
        return (f"H={self.active}/{self.max} | steps={self.steps_here} | "
                f"EMA={ema_str} | no_improve={self.steps_no_improve}")


class BestLevelCheckpointer:
    CKPT_PATH = "/kaggle/working/best_horizon_ckpt.pt"

    def __init__(self, save_path=CKPT_PATH):
        self.save_path           = save_path
        self.best_loss_per_level : Dict[int, float] = {}
        self.best_overall_loss   = float("inf")
        self.best_overall_level  = 0
        self.best_step           = -1
        self._saves              = 0
        self.last_save_step      = -999

    def maybe_save(self, model, level, loss, step):
        is_best = False
        if loss < self.best_loss_per_level.get(level, float("inf")):
            self.best_loss_per_level[level] = loss
            is_best = True

        if loss < self.best_overall_loss:
            self.best_overall_loss  = loss
            self.best_overall_level = level
            self.best_step          = step

        if not is_best:
            return False

        if step - self.last_save_step < 100:
            return True

        self.last_save_step = step

        trainable_state = {
            name: param.detach().cpu().clone()
            for name, param in model.named_parameters()
            if any(name.startswith(pfx) for pfx in TRAINABLE_PREFIXES)
        }
        torch.save({
            "level": level, "loss": loss, "step": step,
            "trainable_state": trainable_state,
            "loss_per_level": dict(self.best_loss_per_level),
        }, self.save_path)
        self._saves += 1

        gc.collect()
        return True

    def restore_best(self, model, device):
        if not os.path.exists(self.save_path):
            print("  ⚠️  No checkpoint found.", flush=True)
            return None
        ckpt = torch.load(self.save_path, map_location=device)
        filtered = {k: v for k, v in ckpt["trainable_state"].items()
                    if k in model.state_dict()}
        model.load_state_dict(filtered, strict=False)
        print(f"  ✅ Restored H={ckpt['level']} loss={ckpt['loss']:.4f} "
              f"from step {ckpt['step']}", flush=True)
        return ckpt

    def summary(self):
        if not self.best_loss_per_level:
            return "no checkpoints saved"
        levels = ", ".join(f"H={l}:{v:.4f}"
                           for l, v in sorted(self.best_loss_per_level.items()))
        return (f"saves={self._saves} | best: H={self.best_overall_level} "
                f"loss={self.best_overall_loss:.4f} @ step {self.best_step} "
                f"| [{levels}]")


def _horizon_grad_norms(model):
    hp = model.horizon_predictor
    groups = {
        "context_gate": list(hp.context_gate.parameters()),
        "cross_attn":   list(hp.cross_attn.parameters()),
        "cross_ffn":    list(hp.cross_ffn.parameters()),
        "causal_attn":  list(hp.causal_attn.parameters()),
        "causal_ffn":   list(hp.causal_ffn.parameters()),
        "proj":         list(hp.proj.parameters()),
        "step_queries": [hp.step_queries],
    }
    result = {}
    for name, params in groups.items():
        grads = [p.grad for p in params if p.grad is not None]
        if not grads:
            result[name] = 0.0; continue
        total_sq = sum(g.detach().float().pow(2).sum().item() for g in grads)
        result[name] = math.sqrt(total_sq)
    return result


def _fmt_norms(norms):
    return "  ".join(f"{k}={v:.3f}"
                     for k, v in sorted(norms.items(), key=lambda x: x[1], reverse=True))


_MAX_VALID_LOSS = 200.0


# ══════════════════════════════════════════════════════════════
#  TRAINING LOOP
# ══════════════════════════════════════════════════════════════

def train_horizon(raw_model, model, tokenizer, corpus, hf_token):
    cfg = Config

    proj_param_names = {
        f"horizon_predictor.proj.{n}"
        for n, _ in raw_model.horizon_predictor.proj.named_parameters()
    }
    proj_params, other_params = [], []
    for name, param in raw_model.named_parameters():
        if param.requires_grad:
            if name in proj_param_names:
                proj_params.append(param)
            else:
                other_params.append(param)

    optimizer = torch.optim.AdamW([
        {"params": other_params, "lr_base": cfg.HORIZON_LR,
         "lr": cfg.HORIZON_LR, "weight_decay": cfg.WEIGHT_DECAY},
        {"params": proj_params,  "lr_base": cfg.HORIZON_LR,
         "lr": cfg.HORIZON_LR, "weight_decay": cfg.PROJ_WEIGHT_DECAY},
    ], eps=1e-4, betas=(0.9, 0.95))

    n_trainable = sum(p.numel() for p in raw_model.parameters() if p.requires_grad)
    print(f"Optimizer: AdamW (β1=0.9, β2=0.95, ε=1e-4) | "
          f"LR={cfg.HORIZON_LR:.1e} | trainable={n_trainable/1e6:.1f}M", flush=True)

    scaler = GradScaler(enabled=cfg.USE_AMP, init_scale=256)
    model.train()

    t0         = time.time()
    deadline   = t0 + cfg.TIME_BUDGET_S - cfg.UPLOAD_MARGIN_S
    last_mid   = t0
    step       = 0
    best_loss  = float("inf")
    consecutive_nans = 0
    MAX_STEPS = 20_000

    recent_losses = collections.deque(maxlen=50)
    spike_ema = None

    n_gpu = torch.cuda.device_count()
    eff_batch = cfg.BATCH_SIZE * n_gpu * cfg.GRAD_ACCUM

    print(f"\n{'='*70}", flush=True)
    print(f"  HORIZON FINETUNE v10 | 28h | {n_gpu}× T4", flush=True)
    print(f"  LR={cfg.HORIZON_LR} | warmup={cfg.WARMUP_STEPS} | plateau_end={cfg.LR_PLATEAU_END}", flush=True)
    print(f"  Adam: β2=0.95 ε=1e-4 | clamp REMOVED | MAX LIMIT {_MAX_VALID_LOSS}", flush=True)
    print(f"  Eff batch={eff_batch} | GRAD_CLIP={cfg.GRAD_CLIP}", flush=True)
    print(f"  Curriculum: gate={cfg.HORIZON_GATE_LOSS} | min_steps={cfg.HORIZON_GATE_MIN_STEPS}", flush=True)
    print(f"  Uploads: every {cfg.MID_UPLOAD_INTERVAL_S//3600}h (Background Threading)", flush=True)
    print(f"  v10 fixes: plateau_end=600, half-LR at warmup boundary, nan threshold=20", flush=True)
    print(f"{'='*70}\n", flush=True)

    curriculum   = HorizonCurriculumTracker(
        max_horizon=cfg.HORIZON,
        gate_loss=cfg.HORIZON_GATE_LOSS,
        min_steps=cfg.HORIZON_GATE_MIN_STEPS,
        ema_alpha=cfg.HORIZON_EMA_ALPHA,
        force_advance_steps=cfg.HORIZON_FORCE_ADVANCE_STEPS,
    )
    checkpointer = BestLevelCheckpointer()

    print(f"  🎓 Starting H=1/{cfg.HORIZON}  gate<{cfg.HORIZON_GATE_LOSS}"
          f"  force-advance={'DISABLED' if cfg.HORIZON_FORCE_ADVANCE_STEPS > 1000 else cfg.HORIZON_FORCE_ADVANCE_STEPS}", flush=True)
    print(f"  💾 BestLevelCheckpointer → {checkpointer.save_path}", flush=True)
    print(f"\nTraining started. Deadline in {(deadline-t0)/3600:.1f} hours...\n", flush=True)

    while time.time() < deadline and step < MAX_STEPS:
        now = time.time()

        if now - last_mid >= cfg.MID_UPLOAD_INTERVAL_S:
            mid_name = f"horizon_mid_step{step}"
            upload_to_hf(raw_model, tokenizer, mid_name, step, hf_token, sync=False)
            last_mid = time.time()

        # ── LR schedule ───────────────────────────────────────
        lr_scale = cyclic_cosine_lr(
            step,
            warmup      = cfg.WARMUP_STEPS,
            plateau_end = cfg.LR_PLATEAU_END,
            cycle_steps = cfg.LR_CYCLE_STEPS,
            base        = 1.0,
            min_frac    = cfg.LR_CYCLE_MIN_FRAC,
        )
        for pg in optimizer.param_groups:
            pg["lr"] = pg["lr_base"] * lr_scale

        # v10 FIX: halve LR at the exact warmup→plateau transition to absorb
        # the first full-LR gradient spike before Adam moment estimates stabilise
        if step == cfg.WARMUP_STEPS:
            for pg in optimizer.param_groups:
                pg["lr"] *= 0.5
            print(f"  [v10] Warmup boundary at step {step}: LR halved to {optimizer.param_groups[0]['lr']:.2e}", flush=True)

        active_h = curriculum.active
        optimizer.zero_grad(set_to_none=True)
        accum_hor, micro_valid = 0.0, 0
        max_logit_this_step = 0.0

        for _micro in range(cfg.GRAD_ACCUM):
            if time.time() >= deadline:
                break
            batch     = get_batch(corpus, cfg.BATCH_SIZE * n_gpu,
                                  cfg.MAX_SEQ_LEN, cfg.HORIZON)
            input_ids = batch[:, :cfg.MAX_SEQ_LEN]
            labels    = batch[:, 1: cfg.MAX_SEQ_LEN + 1]
            hor_tgts  = batch[:, cfg.MAX_SEQ_LEN : cfg.MAX_SEQ_LEN + cfg.HORIZON]

            try:
                with autocast_ctx(enabled=cfg.USE_AMP):
                    hor_loss, logit_max_batch = model(input_ids, labels, hor_tgts, active_h)

                if hor_loss is None or (hor_loss == 0.0).all():
                    del batch, input_ids, labels, hor_tgts; continue

                hor_loss_scalar = hor_loss.mean()
                raw_v = hor_loss_scalar.item()

                if not math.isfinite(raw_v) or raw_v > _MAX_VALID_LOSS or raw_v == 0.0:
                    del hor_loss, hor_loss_scalar, batch, input_ids, labels, hor_tgts
                    continue

                loss = hor_loss_scalar / cfg.GRAD_ACCUM
                scaler.scale(loss).backward()
                accum_hor += raw_v; micro_valid += 1

                with torch.no_grad():
                    logit_max = logit_max_batch.max().item()
                    if logit_max > max_logit_this_step:
                        max_logit_this_step = logit_max

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"\n⚠️  OOM at step {step} micro {_micro}", flush=True)
                    torch.cuda.empty_cache(); gc.collect()
                else:
                    print(f"\n⚠️  RuntimeError: {e}", flush=True)
                break
            except Exception as e:
                print(f"\n⚠️  Exception: {e}", flush=True); break

            del hor_loss, loss, batch, input_ids, labels, hor_tgts

        if micro_valid == 0:
            consecutive_nans += 1
            optimizer.zero_grad(set_to_none=True)
            if consecutive_nans > 20:  # v10 FIX: was 8
                print(f"\n❌ Too many consecutive empty steps — stopping.", flush=True)
                break
            continue

        hor_v = accum_hor / micro_valid

        scaler.unscale_(optimizer)
        module_norms = _horizon_grad_norms(raw_model)
        total_norm   = nn.utils.clip_grad_norm_(
            [p for p in raw_model.parameters() if p.requires_grad],
            cfg.GRAD_CLIP
        )

        has_nan = False
        for p in raw_model.parameters():
            if p.requires_grad and p.grad is not None:
                if not torch.isfinite(p.grad).all():
                    has_nan = True
                    break

        scaler.step(optimizer)
        scaler.update()

        if has_nan:
            consecutive_nans += 1
            optimizer.zero_grad(set_to_none=True)
            if consecutive_nans > 20:  # v10 FIX: was 8
                print(f"\n❌ Too many NaN gradients — stopping.", flush=True)
                break
            continue

        consecutive_nans = 0

        if spike_ema is None:
            spike_ema = hor_v
        else:
            spike_ema = 0.9 * spike_ema + 0.1 * hor_v

        if spike_ema is not None and hor_v > spike_ema * 1.5:
            print(f"  🔍 Spike at step {step}: loss={hor_v:.4f} (EMA={spike_ema:.4f}), max logit={max_logit_this_step:.1f}", flush=True)

        recent_losses.append(hor_v)

        if hor_v < best_loss:
            best_loss = hor_v

        advanced, reason = curriculum.update(hor_v)
        if advanced:
            tag = "📈 gate" if reason == "gate" else "⏭️  force-advance"
            print(
                f"\n  {tag} at step {step}! "
                f"→ horizon now H={curriculum.active}/{cfg.HORIZON}\n",
                flush=True,
            )

        saved = checkpointer.maybe_save(raw_model, active_h, hor_v, step)
        if saved:
            print(f"  💾 New best at H={active_h}: {hor_v:.4f} (step {step})", flush=True)

        elapsed   = time.time() - t0
        remaining = deadline - time.time()

        if step % 10 == 0:
            lr_now  = optimizer.param_groups[0]["lr"]
            mem_gb  = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
            trend   = "↓" if hor_v <= best_loss else "↑"
            ema_str = f"{curriculum.ema_loss:.4f}" if curriculum.ema_loss else "n/a"
            cycle_n = max(1, (max(0, step - cfg.LR_PLATEAU_END)) // cfg.LR_CYCLE_STEPS + 1)
            print(
                f"Step {step:5d} {trend} | H={active_h}/{cfg.HORIZON} EMA={ema_str} | "
                f"loss={hor_v:.4f} best={best_loss:.4f} | "
                f"lr={lr_now:.2e} [c{cycle_n}] | norm={total_norm:.2f} | "
                f"mem={mem_gb:.1f}GB | {elapsed/3600:.2f}h/{(elapsed+remaining)/3600:.2f}h",
                flush=True,
            )

        if step % 50 == 0:
            print(f"  grad breakdown | {_fmt_norms(module_norms)}", flush=True)
            torch.cuda.empty_cache()
            gc.collect()
            try: ctypes.CDLL("libc.so.6").malloc_trim(0)
            except: pass

        if step % 100 == 0 and step > 0 and len(recent_losses) >= 50:
            first_half  = list(recent_losses)[:25]
            second_half = list(recent_losses)[25:]
            avg_first   = sum(first_half)  / len(first_half)
            avg_second  = sum(second_half) / len(second_half)
            improvement = avg_first - avg_second
            print(f"  📊 Rolling improvement over last 50 steps: "
                  f"{improvement:+.4f} ({avg_first:.4f}→{avg_second:.4f})", flush=True)

        step += 1

        if best_loss < 1.0:
            print(f"\n🎯 Target reached! loss = {best_loss:.4f} < 1.0 at step {step}", flush=True)
            break

    print(f"\n{'='*70}", flush=True)
    print(f"  Training done. Steps: {step} | Best loss: {best_loss:.4f}", flush=True)
    print(f"  Curriculum: {curriculum.status()}", flush=True)
    print(f"  Checkpoints: {checkpointer.summary()}", flush=True)
    print(f"{'='*70}\n", flush=True)

    print("🔄 Restoring best checkpoint...", flush=True)
    ckpt = checkpointer.restore_best(raw_model, cfg.DEVICE)
    if ckpt and ckpt["loss"] < best_loss:
        best_loss = ckpt["loss"]
        print(f"  📊 Updated best loss to {best_loss:.4f}", flush=True)


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════

def kaggle_horizon_finetune():
    print("\n" + "="*70, flush=True)
    print("  IOv52-HST — HORIZON PREDICTOR FINETUNE v10", flush=True)
    print("  28 hours | 2× T4 DataParallel", flush=True)
    print("  LR=3e-5, warmup=200, plateau_end=600, clamp REMOVED", flush=True)
    print("="*70, flush=True)

    if not HF_TOKEN:
        raise RuntimeError("No HF token! Add KAGGLE_HF_TOKEN to Kaggle Secrets.")

    n_gpu = torch.cuda.device_count()
    print(f"GPUs available: {n_gpu}", flush=True)

    total_vram = 0.0
    if torch.cuda.is_available():
        for i in range(n_gpu):
            free, total = torch.cuda.mem_get_info(i)
            total_vram += total / 1e9
            print(f"  GPU {i}: {free/1e9:.1f}GB free / {total/1e9:.1f}GB total", flush=True)

    Config.BATCH_SIZE = 2 if total_vram >= 28.0 else 1
    print(f"  Batch per step = {Config.BATCH_SIZE}×{n_gpu}GPU×{Config.GRAD_ACCUM}acc"
          f" = {Config.BATCH_SIZE * n_gpu * Config.GRAD_ACCUM} eff", flush=True)

    torch.cuda.empty_cache(); gc.collect()

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    print("\nBuilding model...", flush=True)
    raw_model = IOHSTCombined(Config)
    n_params  = sum(p.numel() for p in raw_model.parameters()) / 1e6
    print(f"Total parameters: {n_params:.1f}M", flush=True)

    load_checkpoint(raw_model, Config.HF_REPO_ID,
                    Config.CHECKPOINT_SUBFOLDER, HF_TOKEN)

    raw_model = raw_model.to(Config.DEVICE)
    raw_model = freeze_all_except_horizon(raw_model)

    if n_gpu > 1:
        print(f"\n⚡ Wrapping with nn.DataParallel across {n_gpu} GPUs", flush=True)
        model = nn.DataParallel(raw_model)
    else:
        model = raw_model

    mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    print(f"GPU memory after load: {mem:.2f} GB", flush=True)

    corpus = build_corpus(tokenizer, Config.DEVICE, max_tokens=8_000_000)

    train_horizon(raw_model, model, tokenizer, corpus, HF_TOKEN)

    print(f"\n📤 Final upload as '{Config.FINAL_UPLOAD_NAME}'...", flush=True)
    success = upload_to_hf(
        raw_model, tokenizer,
        Config.FINAL_UPLOAD_NAME, step=-1,
        hf_token=HF_TOKEN, repo_id=Config.HF_REPO_ID,
        sync=True
    )

    if success:
        print(f"\n🎉 SUCCESS — {Config.HF_REPO_ID}/{Config.FINAL_UPLOAD_NAME}", flush=True)
    else:
        torch.save(raw_model.state_dict(), "/kaggle/working/FULL_model.pt")
        print("⚠️  Upload failed — saved locally.", flush=True)

    print("\nDone.", flush=True)


KAGGLE_RUN = os.environ.get("KAGGLE_KERNEL_RUN_TYPE") is not None
if __name__ == "__main__" or KAGGLE_RUN:
    kaggle_horizon_finetune()
