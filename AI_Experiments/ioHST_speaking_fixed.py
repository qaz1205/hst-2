# ============================================================
#  IOv52-HST — SPEAKING FINETUNING  (FIXED)
# ============================================================
#
#  BUG-1  ChunkDecoderWithCache: chunk_idx always 0 during training
#         (no cache → past_len=0 → chunk_idx=0 for every token).
#         v1 fix expanded chunk_emb to [B,S,D] with repeated rows —
#         that made cross-attention degenerate (identical K/V ⇒ zero
#         gradient through attention weights ⇒ chunk encoder dead).
#         FIX v2: pass chunk_embeddings directly as [B, n_chunks, D].
#         Each token now attends over all distinct chunk representations.
#
#  BUG-2  Weight-tying on dead lm_head.
#         IOHSTCombined tied self.lm_head (never called).
#         FIX: tie chunk_decoder.lm_head.weight = token_emb.weight.
#
#  BUG-3  ln_f defined but never applied.
#         FIX: apply ln_f to enc_input before the chunk decoder.
#
#  BUG-4  LR imbalance — chunk_decoder at 1e-4 vs chunk_encoder at
#         1e-5 and top_blocks at 5e-6.  Decoder saturates the enc_input
#         shortcut in ~300 steps; chunk encoder and top blocks learn
#         33-70× slower and never catch up → plateau at loss ~8.3.
#         FIX: chunk_encoder → 5e-5, top_blocks → 3e-5 (own groups).
#
#  OPT-1  Duplicate params in optimizer groups (original _group had no
#         tied-weight guard).  FIX: _TIED_IDS exclusion set.
#
# ============================================================

import subprocess, sys, os, gc, math, time, random, warnings, shutil, glob
from typing import Dict, List, Any, Optional, Tuple

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(line_buffering=True)

os.environ["PYDEVD_DISABLE_FILE_VALIDATION"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ── HF Token ─────────────────────────────────────────────────────────────────
HF_TOKEN = ""

if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") is not None:
    try:
        from kaggle_secrets import UserSecretsClient
        _us = UserSecretsClient()
        for _n in ["KAGGLE_HF_TOKEN", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"]:
            try:
                HF_TOKEN = _us.get_secret(_n) or ""
                if HF_TOKEN: print(f"✓ HF token: Kaggle / '{_n}'"); break
            except Exception: pass
    except ImportError: pass
else:
    try:
        from google.colab import userdata
        HF_TOKEN = userdata.get("HF_TOKEN") or ""
        if HF_TOKEN: print("✓ HF token: Colab Secrets")
    except Exception: pass

if not HF_TOKEN:
    HF_TOKEN = (os.environ.get("KAGGLE_HF_TOKEN") or
                os.environ.get("HF_TOKEN") or
                os.environ.get("HUGGING_FACE_HUB_TOKEN") or "")
    if HF_TOKEN: print("✓ HF token: environment variable")

if not HF_TOKEN:
    raise RuntimeError("No HF token found! Paste into HF_TOKEN = '' above")

os.environ["HF_TOKEN"] = HF_TOKEN
os.environ["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN

# ── Dependencies ─────────────────────────────────────────────────────────────
def _pip(*pkgs, timeout=300):
    r = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "--no-warn-conflicts", *pkgs],
        capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"pip failed:\n{r.stderr[-2000:]}")
    print(f"  ✓ {' '.join(pkgs)}")

print("Installing dependencies...")
_pip("transformers>=4.40.0", "datasets>=2.19.0", "accelerate>=0.30.0", "huggingface_hub>=0.23.0")
print("Dependencies ready ✓\n")

# ── Imports ───────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2TokenizerFast
from datasets import load_dataset
from huggingface_hub import HfApi, hf_hub_download
import transformers, logging as _log
transformers.logging.set_verbosity_error()
_log.getLogger("transformers").setLevel(_log.ERROR)
_log.getLogger("datasets").setLevel(_log.ERROR)
warnings.filterwarnings("ignore")

print(f"PyTorch {torch.__version__} | CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(i)
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}"
              f"  ({free/1e9:.1f}/{total/1e9:.1f} GB free)")

# ── Config ────────────────────────────────────────────────────────────────────
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
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

HF_REPO        = "thagnitti/io"
CHECKPOINT_SUB = "checkpoint_step_5000"
SAVE_SUBFOLDER = "speaking_v1"

# ── Training hypers ───────────────────────────────────────────────────────────
TOTAL_STEPS    = 3000
EVAL_EVERY     = 100
SAVE_EVERY     = 500
BATCH_SIZE     = 2
GRAD_ACCUM     = 4
MAX_SEQ_LEN    = Config.MAX_SEQ_LEN
USE_AMP        = False     # fp32 required — SSM/Hebbian unstable under AMP
MAX_GRAD_NORM  = 0.5
HP_LOSS_WEIGHT = 0.05
WARMUP_STEPS   = 200
WEIGHT_DECAY   = 0.01

# BUG-4 FIX: separate LRs so hierarchical path trains at same rate as decoder.
# Old code had chunk_enc at 1e-5 and top_blocks at 5e-6 while chunk_decoder
# was at 1e-4 — decoder saturated enc_input shortcut in ~300 steps, chunk
# encoder received 33-70x weaker effective updates and never caught up.
LR_CHUNK_DECODER = 1e-4   # decoder output head — learns fastest
LR_CHUNK_ENC     = 5e-5   # chunk encoder — must keep pace with decoder
LR_TOP           = 3e-5   # top transformer blocks — critical for h_top quality
LR_HP            = 1e-5
LR_CIF           = 2e-5
LR_DIAMOND       = 2e-5
LR_BACKBONE      = 5e-6   # bottom blocks only
LR_EMBED         = 2e-6

random.seed(42); torch.manual_seed(42)
print(f"\nDevice: {Config.DEVICE}")
print(f"Steps: {TOTAL_STEPS}  Eval every: {EVAL_EVERY}  Save every: {SAVE_EVERY}")
print(f"Effective batch: {BATCH_SIZE}×{GRAD_ACCUM}={BATCH_SIZE*GRAD_ACCUM}")
print(f"Checkpoint: {HF_REPO}/{CHECKPOINT_SUB}  →  {HF_REPO}/{SAVE_SUBFOLDER}/\n")

# ════════════════════════════════════════════════════════════
#  ARCHITECTURE
# ════════════════════════════════════════════════════════════

class SelectiveSSM(nn.Module):
    def __init__(self, d_model, d_state=8, d_conv=4, expand=1):
        super().__init__()
        self.d_model = d_model; self.d_inner = d_model * expand; self.d_state = d_state
        self.in_proj  = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv     = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=d_conv,
                                  padding=d_conv - 1, groups=self.d_inner, bias=True)
        self.x_proj   = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)
        self.dt_proj  = nn.Linear(1, self.d_inner, bias=True)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).expand(self.d_inner, -1)
        self.log_A = nn.Parameter(torch.log(A)); self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False); self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        residual = x; B, L, D = x.shape
        xz = self.in_proj(x); x_main, z = xz.chunk(2, dim=-1)
        x_conv = self.conv(x_main.transpose(1, 2))[:, :, :L].transpose(1, 2)
        x_act  = F.silu(x_conv); ssm_p = self.x_proj(x_act)
        B_mat, C_mat, dt_raw = ssm_p.split([self.d_state, self.d_state, 1], dim=-1)
        dt = F.softplus(self.dt_proj(dt_raw)); A = -torch.exp(self.log_A)
        dA = torch.exp(dt.unsqueeze(-1) * A); dB = dt.unsqueeze(-1) * B_mat.unsqueeze(2)
        cum_A = torch.exp(torch.cumsum(torch.log(dA.clamp(min=1e-8)), dim=1))
        contrib = dB * x_act.unsqueeze(-1)
        h_approx = torch.cumsum(contrib / cum_A.clamp(min=1e-8), dim=1) * cum_A
        y = (h_approx * C_mat.unsqueeze(2)).sum(-1); out = y + self.D * x_act
        return self.norm(self.out_proj(out * F.silu(z)) + residual)

class DiamondMixer(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.split_proj = nn.Linear(d_model, d_model * 2)
        self.z_net  = nn.Sequential(nn.Linear(d_model, d_model*4), nn.GELU(), nn.Linear(d_model*4, d_model))
        self.w_net  = nn.Sequential(nn.Linear(d_model, d_model*4), nn.GELU(), nn.Linear(d_model*4, d_model))
        self.merge_proj = nn.Linear(d_model * 2, d_model); self.norm = nn.LayerNorm(d_model)

    def forward(self, u):
        x, y = self.split_proj(u).chunk(2, dim=-1)
        return self.norm(u + self.merge_proj(torch.cat([self.z_net(x+y), self.w_net(y-x)], dim=-1)))

class HebbianFastWeights(nn.Module):
    def __init__(self, d_model, lambda_decay=0.95):
        super().__init__()
        self.lambda_decay = lambda_decay
        self.qkv = nn.Linear(d_model, d_model * 3, bias=False); self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        B, S, D = x.shape; qkv = self.qkv(x).reshape(B, S, 3, D).permute(2, 0, 1, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]
        kv = torch.einsum('bsd,bse->bde', k, v) * self.lambda_decay
        out = torch.einsum('bsd,bde->bse', q, kv)
        return self.norm(x + out * torch.sigmoid((q * k).sum(dim=-1, keepdim=True)))

class CIFModule(nn.Module):
    def __init__(self, d_model, threshold=0.5):
        super().__init__()
        self.chunk_size = max(1, round(1.0 / threshold))
        self.weight_proj = nn.Linear(d_model, 1)
        nn.init.constant_(self.weight_proj.bias, -4.85); nn.init.zeros_(self.weight_proj.weight)
        self.value_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, L, D = x.shape; cs = self.chunk_size
        alphas = torch.sigmoid(self.weight_proj(x)).squeeze(-1); values = self.value_proj(x)
        pad = (cs - L % cs) % cs
        if pad: values = F.pad(values, (0,0,0,pad)); alphas_p = F.pad(alphas, (0, pad))
        else:   alphas_p = alphas
        n_chunks = (L + pad) // cs
        v_chunks = values.reshape(B, n_chunks, cs, D); a_chunks = alphas_p.reshape(B, n_chunks, cs).unsqueeze(-1)
        return (v_chunks * a_chunks).sum(2) / a_chunks.sum(2).clamp(min=1e-3), alphas

class SelfAttentionWithCache(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads; self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, d_model * 3, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, layer_past=None):
        B, S, D = x.shape
        qkv = self.qkv(x).reshape(B, S, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        if layer_past is not None: k = torch.cat([layer_past[0], k], dim=2); v = torch.cat([layer_past[1], v], dim=2)
        present = (k, v)
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=(layer_past is None))
        return self.out_proj(attn.transpose(1, 2).contiguous().view(B, S, D)), present

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, use_diamond_ffn=False, dropout=0.1):
        super().__init__()
        self.attn = SelfAttentionWithCache(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model); self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout); self.use_diamond = use_diamond_ffn
        if use_diamond_ffn:
            self.ffn = DiamondMixer(d_model)
        else:
            self.ff1 = nn.Linear(d_model, 4*d_model); self.ff2 = nn.Linear(4*d_model, d_model)
            self.act = nn.GELU(approximate='tanh')

    def forward(self, x, layer_past=None):
        attn_out, present = self.attn(self.norm1(x), layer_past)
        x = x + self.drop(attn_out)
        if self.use_diamond: x = self.ffn(x)
        else: x = x + self.drop(self.ff2(self.act(self.ff1(self.norm2(x)))))
        return x, present

class AdaptiveBlock(nn.Module):
    def __init__(self, d_model, n_heads, use_ssm=False, use_hebbian=False):
        super().__init__()
        self.block = TransformerBlock(d_model, n_heads)
        self.use_ssm = use_ssm; self.use_hebbian = use_hebbian
        if use_ssm:     self.ssm     = SelectiveSSM(d_model, d_state=Config.SSM_D_STATE, expand=Config.SSM_EXPAND)
        if use_hebbian: self.hebbian = HebbianFastWeights(d_model)
        self.confidence_head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(),
                                             nn.Linear(d_model, 1), nn.Sigmoid())

    def forward(self, x, layer_past=None):
        x_out, present = self.block(x, layer_past)
        if self.use_ssm:     x_out = self.ssm(x_out)
        if self.use_hebbian: x_out = self.hebbian(x_out)
        conf = (self.confidence_head(x_out.transpose(1,2)).mean(0) if x_out.size(1) > 1
                else x_out.new_tensor([0.0]))
        return x_out, conf, present

class DiamondCrossAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads; self.head_dim = d_model // n_heads
        self.q = nn.Linear(d_model, d_model, bias=False); self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False); self.out = nn.Linear(d_model, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model); self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, top_h, bottom_h):
        B, S, D = top_h.shape
        q = self.q(top_h).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k(bottom_h).view(B, -1, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v(bottom_h).view(B, -1, self.n_heads, self.head_dim).transpose(1, 2)
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        return self.norm(top_h + torch.tanh(self.gate) * self.out(attn.transpose(1,2).contiguous().view(B,S,D)))

class ChunkEncoder(nn.Module):
    def __init__(self, d_model, chunk_size=128, n_heads=8, n_layers=2):
        super().__init__()
        self.chunk_size = chunk_size
        enc = nn.TransformerEncoderLayer(d_model, n_heads, d_model*4, batch_first=True, dropout=0.0)
        self.local_encoder = nn.TransformerEncoder(enc, num_layers=n_layers)
        self.pooling_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pooling_attn  = nn.MultiheadAttention(d_model, n_heads, batch_first=True)

    def forward(self, token_embeddings):
        B, total, D = token_embeddings.shape; cs = self.chunk_size; n_chunks = total // cs
        chunks = token_embeddings[:, :n_chunks*cs, :].view(B*n_chunks, cs, D)
        encoded = self.local_encoder(chunks)
        query = self.pooling_query.expand(B*n_chunks, -1, -1)
        pooled, _ = self.pooling_attn(query, encoded, encoded)
        return pooled.view(B, n_chunks, D)

class TransformerDecoderLayerWithCache(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.self_attn = SelfAttentionWithCache(d_model, n_heads)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.linear1 = nn.Linear(d_model, 4*d_model); self.linear2 = nn.Linear(4*d_model, d_model)
        self.norm1 = nn.LayerNorm(d_model); self.norm2 = nn.LayerNorm(d_model); self.norm3 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, tgt, memory, self_attn_past=None, cross_attn_past=None):
        sa_out, sa_present = self.self_attn(self.norm1(tgt), layer_past=self_attn_past)
        tgt = tgt + self.drop(sa_out)
        if cross_attn_past is not None:
            ca_out, _ = self.cross_attn(self.norm2(tgt), cross_attn_past[0], cross_attn_past[1])
            ca_present = cross_attn_past
        else:
            ca_out, _ = self.cross_attn(self.norm2(tgt), memory, memory); ca_present = (memory, memory)
        tgt = tgt + self.drop(ca_out)
        return tgt + self.drop(self.linear2(self.drop(F.gelu(self.linear1(self.norm3(tgt)))))), sa_present, ca_present

class ChunkDecoderWithCache(nn.Module):
    def __init__(self, d_model, vocab_size, chunk_size=128, n_heads=8, n_layers=2):
        super().__init__()
        self.chunk_size = chunk_size
        self.pos_embedding = nn.Embedding(chunk_size, d_model)
        self.layers = nn.ModuleList([TransformerDecoderLayerWithCache(d_model, n_heads) for _ in range(n_layers)])
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, chunk_embeddings, target_token_embeddings, cache=None):
        B, S, D = target_token_embeddings.shape; device = target_token_embeddings.device
        past_len = cache[0][0][0].size(2) if cache else 0
        positions = torch.arange(past_len, past_len+S, dtype=torch.long, device=device) % self.chunk_size
        tgt = target_token_embeddings + self.pos_embedding(positions)
        new_cache = []
        for i, layer in enumerate(self.layers):
            sa_past, ca_past = cache[i] if cache else (None, None)
            # BUG-1 FIX v2: pass chunk_embeddings directly as [B, n_chunks, D].
            # Original always used chunk_idx=0 (wrong).
            # v1 fix expanded to [B,S,D] with repeated rows — degenerate K/V,
            # zero gradient through attention weights, chunk encoder dead.
            # Using the full [B, n_chunks, D] tensor gives n_chunks distinct
            # K/V pairs so every token can genuinely attend to each chunk.
            memory = chunk_embeddings  # [B, n_chunks, D]
            tgt, sa_present, ca_present = layer(tgt, memory, sa_past, ca_past)
            new_cache.append((sa_present, ca_present))
        return self.lm_head(tgt), new_cache

class FullLatticeFieldAnalyzer(nn.Module):
    def __init__(self, max_seq_len=8192):
        super().__init__()
        spine = [0, 2, 4]
        while True:
            nv = 2*spine[-1] + 2*spine[-2] + 2*spine[-3]
            if nv >= max_seq_len: break
            spine.append(nv)
        self.register_buffer('spine', torch.tensor(spine, dtype=torch.long))
        self.lattice_structure = {}; self._non_spine_cache: Dict[int, Any] = {}
        for pos in spine:
            if pos < max_seq_len: self.lattice_structure[pos] = self._analyze_position(pos)

    def get_structure(self, pos):
        if pos in self.lattice_structure: return self.lattice_structure[pos]
        if pos in self._non_spine_cache:  return self._non_spine_cache[pos]
        s = self._analyze_non_spine(pos); self._non_spine_cache[pos] = s; return s

    def _analyze_position(self, pos):
        levels = {0: [pos]}; visited = {pos}; current = [pos]; level = 0
        while current and level < 10:
            nxt = set()
            for node in current:
                for anc in self._get_ancestors(node):
                    if anc not in visited and anc >= 0: visited.add(anc); nxt.add(anc)
            current = list(nxt); level += 1
            if current: levels[level] = current.copy()
        md = max(levels.keys()) if levels else 0
        return {'levels': levels, 'path_counts': self._compute_path_counts(pos, levels, md),
                'total_ancestors': len(visited)-1, 'max_depth': md}

    def _get_ancestors(self, pos):
        try:
            idx = (self.spine == pos).nonzero(as_tuple=True)[0].item()
            if idx >= 3: return [self.spine[idx-1].item(), self.spine[idx-2].item(), self.spine[idx-3].item()]
        except Exception: pass
        return []

    def _analyze_non_spine(self, pos):
        left = self.spine[self.spine < pos]; ancs = [left[-1].item()] if len(left) > 0 else []
        return {'levels': {0: [pos], 1: ancs}, 'path_counts': {a: 1 for a in ancs},
                'total_ancestors': len(ancs), 'max_depth': 1}

    def _compute_path_counts(self, pos, levels, max_depth):
        pc = {pos: 1}
        for level in sorted(levels.keys(), reverse=True):
            for node in levels[level]:
                if node == pos: continue
                if level == max_depth: pc[node] = 1; continue
                count = sum(pc.get(child, 0) for child in levels.get(level+1, [])
                            if node in self._get_ancestors(child))
                if level != 0: pc[node] = count
        pc.pop(pos, None); return pc

class MultiLevelLatticeProcessor(nn.Module):
    def __init__(self, d_model, max_seq_len):
        super().__init__()
        self.analyzer = FullLatticeFieldAnalyzer(max_seq_len)
        self.level_transforms = nn.ModuleList([
            nn.Sequential(nn.Linear(d_model, d_model), nn.LayerNorm(d_model), nn.GELU(),
                          nn.Linear(d_model, d_model)) for _ in range(10)])
        self.level_attention = nn.MultiheadAttention(d_model, 4, batch_first=True)
        self.fusion = nn.Sequential(nn.Linear(d_model*2, d_model), nn.LayerNorm(d_model))

    def forward(self, x):
        B, S, D = x.shape; spine = self.analyzer.spine; rel_spine = spine[spine < S]; updates = {}
        for sp in rel_spine:
            pos = sp.item()
            if pos < 3: continue
            structure = self.analyzer.get_structure(pos)
            if structure is None: continue
            level_features = []
            for level in range(structure['max_depth']+1):
                if level == 0 or level not in structure['levels']: continue
                level_h, total_w = [], 0.0
                for node in structure['levels'][level]:
                    if node < S:
                        w = structure['path_counts'].get(node, 1)
                        level_h.append(x[:, node, :] * w); total_w += w
                if level_h and total_w > 0:
                    feat = torch.stack(level_h, dim=1).sum(1) / total_w
                    level_features.append(self.level_transforms[level](feat))
            if not level_features: continue
            stack = torch.stack(level_features, dim=1)
            attended, _ = self.level_attention(x[:, pos:pos+1, :], stack, stack)
            updates[pos] = self.fusion(torch.cat([attended.squeeze(1), x[:, pos, :]], dim=-1))
        if not updates: return x
        slices, last = [], 0
        for pos in sorted(updates.keys()):
            if pos > last: slices.append(x[:, last:pos, :])
            slices.append(updates[pos].unsqueeze(1)); last = pos + 1
        if last < S: slices.append(x[:, last:S, :])
        return torch.cat(slices, dim=1)

class PathWeightedLatticeCore(nn.Module):
    def __init__(self, d_model, max_seq_len):
        super().__init__()
        self.analyzer        = FullLatticeFieldAnalyzer(max_seq_len)
        self.path_weight_net = nn.Sequential(nn.Linear(1, 64), nn.ReLU(), nn.Linear(64, 1), nn.Softplus())
        self.message_fn      = nn.Sequential(nn.Linear(d_model*2, d_model), nn.LayerNorm(d_model), nn.GELU())
        self.aggregate_fn    = nn.Sequential(nn.Linear(d_model, d_model), nn.Tanh())
        self.aggregate_attn  = nn.Linear(d_model, 1)
        self.update_gate     = nn.Sequential(nn.Linear(d_model*2, d_model), nn.Sigmoid())

    def forward(self, x):
        B, S, D = x.shape; spine = self.analyzer.spine; rel_spine = spine[spine < S]; updates = {}
        for sp in rel_spine:
            pos = sp.item()
            if pos < 3: continue
            structure = self.analyzer.get_structure(pos)
            if structure is None or structure['total_ancestors'] == 0: continue
            ancs, pcounts = [], []
            for level in structure['levels']:
                if level > 0:
                    for anc in structure['levels'][level]:
                        if anc < S: ancs.append(anc); pcounts.append(structure['path_counts'].get(anc, 1))
            if not ancs: continue
            pct = torch.tensor(pcounts, device=x.device).view(-1,1).float()
            pw = self.path_weight_net(pct).squeeze()
            msgs = [self.message_fn(torch.cat([x[:, a, :], x[:, pos, :]], dim=-1)) for a in ancs]
            ms = torch.stack(msgs, dim=1)
            if pw.dim() == 0: ms = ms * pw.view(1,1,1).expand(B,-1,D)
            else:              ms = ms * pw.view(1,-1,1).expand(B,-1,D)
            proj = self.aggregate_fn(ms); weight = torch.softmax(self.aggregate_attn(proj), dim=1)
            agg = (proj * weight).sum(dim=1)
            gate = self.update_gate(torch.cat([agg, x[:, pos, :]], dim=-1))
            updates[pos] = gate * agg + (1 - gate) * x[:, pos, :]
        if not updates: return x
        slices, last = [], 0
        for pos in sorted(updates.keys()):
            if pos > last: slices.append(x[:, last:pos, :])
            slices.append(updates[pos].unsqueeze(1)); last = pos + 1
        if last < S: slices.append(x[:, last:S, :])
        return torch.cat(slices, dim=1)

class CompleteLatticeCore(nn.Module):
    def __init__(self, d_model, max_seq_len):
        super().__init__()
        self.multi_level   = MultiLevelLatticeProcessor(d_model, max_seq_len)
        self.path_weighted = PathWeightedLatticeCore(d_model, max_seq_len)
        self.meta_fusion   = nn.Sequential(
            nn.Linear(d_model*3, d_model*2), nn.LayerNorm(d_model*2),
            nn.GELU(), nn.Linear(d_model*2, d_model))

    def forward(self, x):
        return self.meta_fusion(torch.cat([x, self.multi_level(x), self.path_weighted(x)], dim=-1))

class HarmonicHorizonPredictor(nn.Module):
    def __init__(self, d_model: int, vocab_size: int, horizon: int = 8, n_heads: int = 10):
        super().__init__()
        self.horizon = horizon; d_mid = d_model // 2

        self.context_gate = nn.Sequential(nn.Linear(d_model * 2, d_model, bias=True), nn.Sigmoid())
        self.context_norm = nn.LayerNorm(d_model)
        self.step_queries = nn.Parameter(torch.randn(1, horizon, d_model) * 0.02)

        self.cross_attn      = nn.MultiheadAttention(d_model, n_heads, batch_first=True, dropout=0.0)
        self.cross_norm      = nn.LayerNorm(d_model)
        self.cross_ffn       = nn.Sequential(nn.Linear(d_model, d_model*4), nn.GELU(),
                                             nn.Linear(d_model*4, d_model))
        self.cross_ffn_norm  = nn.LayerNorm(d_model)

        self.causal_attn     = nn.MultiheadAttention(d_model, n_heads, batch_first=True, dropout=0.0)
        self.causal_norm     = nn.LayerNorm(d_model)
        self.causal_ffn      = nn.Sequential(nn.Linear(d_model, d_model*4), nn.GELU(),
                                             nn.Linear(d_model*4, d_model))
        self.causal_ffn_norm = nn.LayerNorm(d_model)

        self.proj = nn.Sequential(
            nn.Linear(d_model, d_mid), nn.GELU(), nn.Linear(d_mid, d_mid), nn.LayerNorm(d_mid),
        )
        self.prediction_head = nn.Linear(d_mid, vocab_size, bias=False)
        self.confidence_head = nn.Sequential(
            nn.Linear(d_model, 64), nn.GELU(), nn.Linear(64, 1), nn.Sigmoid())

        causal_mask = torch.triu(torch.full((horizon, horizon), float('-inf')), diagonal=1)
        self.register_buffer('causal_mask', causal_mask)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.ndim == 2: x = x.unsqueeze(1)
        B, n_chunks, D = x.shape
        x_last = x[:, -1, :]; x_mean = x.mean(dim=1)
        gate   = self.context_gate(torch.cat([x_last, x_mean], dim=-1))
        ctx    = self.context_norm(x_last + gate * x_mean)
        memory = torch.cat([x, ctx.unsqueeze(1)], dim=1)

        queries   = self.step_queries.expand(B, -1, -1)
        ca_out, _ = self.cross_attn(queries, memory, memory)
        step_h    = self.cross_norm(queries + ca_out)
        step_h    = self.cross_ffn_norm(step_h + self.cross_ffn(step_h))

        ar_out, _ = self.causal_attn(step_h, step_h, step_h, attn_mask=self.causal_mask)
        step_h    = self.causal_norm(step_h + ar_out)
        step_h    = self.causal_ffn_norm(step_h + self.causal_ffn(step_h))

        logits     = self.prediction_head(self.proj(step_h))
        confidence = self.confidence_head(step_h).squeeze(-1)
        return logits, confidence

class IOHSTCombined(nn.Module):
    def __init__(self, cfg=Config):
        super().__init__()
        self.cfg = cfg; self.n_bottom = cfg.N_LAYERS // 2
        n_chunks = cfg.MAX_SEQ_LEN // cfg.CHUNK_SIZE

        self.token_emb = nn.Embedding(cfg.VOCAB_SIZE, cfg.D_MODEL)
        self.pos_emb   = nn.Embedding(cfg.MAX_SEQ_LEN * 2, cfg.D_MODEL)
        self.bottom    = nn.ModuleList([
            AdaptiveBlock(cfg.D_MODEL, cfg.N_HEADS,
                          use_ssm=(i < cfg.N_MAMBA_LAYERS),
                          use_hebbian=(i < cfg.N_MAMBA_LAYERS))
            for i in range(self.n_bottom)])
        self.cif           = CIFModule(cfg.D_MODEL, cfg.CIF_THRESHOLD)
        self.chunk_encoder = ChunkEncoder(cfg.D_MODEL, chunk_size=cfg.CHUNK_SIZE,
                                          n_heads=max(1, cfg.N_HEADS // 2),
                                          n_layers=cfg.N_CHUNK_ENC_LAYERS)
        self.lattice   = CompleteLatticeCore(cfg.D_MODEL, n_chunks)
        self.diamond   = DiamondCrossAttention(cfg.D_MODEL, cfg.N_HEADS)
        self.top       = nn.ModuleList([
            TransformerBlock(cfg.D_MODEL, cfg.N_HEADS, use_diamond_ffn=(i % 2 == 1))
            for i in range(cfg.N_TOP_LAYERS)])
        self.chunk_decoder = ChunkDecoderWithCache(
            cfg.D_MODEL, cfg.VOCAB_SIZE,
            chunk_size=cfg.CHUNK_SIZE,
            n_heads=max(1, cfg.N_HEADS // 2),
            n_layers=cfg.N_CHUNK_DEC_LAYERS)
        self.horizon_predictor = HarmonicHorizonPredictor(
            cfg.D_MODEL, cfg.VOCAB_SIZE, cfg.HORIZON, n_heads=cfg.N_HEADS)
        self.ln_f = nn.LayerNorm(cfg.D_MODEL)
        # BUG-2 FIX: original tied self.lm_head (dead code, never called).
        # Tie the head that actually produces logits.
        # NOTE: do NOT add self.lm_head here — it was dead and is removed.
        self.chunk_decoder.lm_head.weight = self.token_emb.weight

    def forward(self, input_ids, labels=None):
        B, S    = input_ids.shape
        pos_ids = torch.arange(S, dtype=torch.long, device=input_ids.device)
        h       = self.token_emb(input_ids) + self.pos_emb(pos_ids)

        for block in self.bottom:
            h, _, _ = block(h)
        bottom_h = h

        if S > 1:
            h_cif_raw, _ = self.cif(h)
            h_cif_up = F.interpolate(h_cif_raw.transpose(1,2), size=S,
                                     mode='linear', align_corners=False).transpose(1,2)
            h = bottom_h + 0.1 * h_cif_up
        else:
            h = bottom_h

        n_chunks   = max(1, S // self.cfg.CHUNK_SIZE)
        target_len = n_chunks * self.cfg.CHUNK_SIZE
        enc_input  = (h[:, :target_len, :] if S >= target_len
                      else F.pad(h.transpose(1,2), (0, target_len-S)).transpose(1,2))

        chunk_emb  = self.chunk_encoder(enc_input)
        h_lattice  = self.lattice(chunk_emb)
        h_bridged  = self.diamond(h_lattice, bottom_h)
        h_top      = h_bridged
        for blk in self.top:
            h_top, _ = blk(h_top)

        # BUG-3 FIX: apply ln_f before the output head.
        # Raw hidden states fed directly to lm_head cause unstable logit scales.
        enc_input_norm = self.ln_f(enc_input)
        logits, _ = self.chunk_decoder(h_top, enc_input_norm)
        if logits.shape[1] > S:   logits = logits[:, :S, :]
        elif logits.shape[1] < S: logits = F.pad(logits, (0,0,0, S-logits.shape[1]))
        logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)

        horizon_logits, _ = self.horizon_predictor(h_top.float())

        result: Dict[str, Any] = {'logits': logits}

        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            lm_loss = F.cross_entropy(
                shift_logits.view(-1, self.cfg.VOCAB_SIZE),
                shift_labels.view(-1), ignore_index=-100)
            result['loss'] = lm_loss

            if S > self.cfg.HORIZON:
                h_targets = labels[:, -self.cfg.HORIZON:].contiguous()
                h_logits  = horizon_logits[:, :h_targets.shape[1], :]
                hp_loss   = F.cross_entropy(
                    h_logits.reshape(-1, self.cfg.VOCAB_SIZE).float(),
                    h_targets.reshape(-1), ignore_index=-100, label_smoothing=0.05)
                result['hp_loss'] = hp_loss
            else:
                result['hp_loss'] = None

        return result

# ════════════════════════════════════════════════════════════
#  fp32 wrapper
# ════════════════════════════════════════════════════════════

def wrap_fp32(module):
    orig = module.forward
    def _fwd(*args, **kwargs):
        fa = tuple(a.float() if isinstance(a, torch.Tensor) else a for a in args)
        fk = {k: v.float() if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()}
        in_dtype = next((a.dtype for a in args if isinstance(a, torch.Tensor)), torch.float32)
        with torch.amp.autocast('cuda', enabled=False):
            out = orig(*fa, **fk)
        if isinstance(out, torch.Tensor): return out.to(in_dtype)
        if isinstance(out, (tuple, list)):
            return type(out)(t.to(in_dtype) if isinstance(t, torch.Tensor) else t for t in out)
        return out
    module.forward = _fwd

# ════════════════════════════════════════════════════════════
#  Build model
# ════════════════════════════════════════════════════════════

print("Building model ...")
model = IOHSTCombined(Config).to(Config.DEVICE)
print(f"  Parameters: {sum(p.numel() for p in model.parameters())/1e9:.2f}B")

_FREEZE = [model.lattice]
for blk in model.bottom:
    if hasattr(blk, 'ssm'):     _FREEZE.append(blk.ssm)
    if hasattr(blk, 'hebbian'): _FREEZE.append(blk.hebbian)

for m in _FREEZE:
    m.float()
    for p in m.parameters(): p.requires_grad_(False)
    wrap_fp32(m)

n_ssm = sum(hasattr(b,'ssm') for b in model.bottom)
n_heb = sum(hasattr(b,'hebbian') for b in model.bottom)
print(f"  Frozen (fp32): lattice + {n_ssm} SSM + {n_heb} Hebbian blocks")
print(f"  Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e9:.2f}B params")

# ════════════════════════════════════════════════════════════
#  Checkpoint loader
# ════════════════════════════════════════════════════════════

def load_checkpoint(model, repo_id, subfolder, token, label):
    print(f"\n📥 Downloading {label} ...")
    try:
        local_path = hf_hub_download(
            repo_id=repo_id, filename=f"{subfolder}/model.pt",
            repo_type='model', token=token)
    except Exception as e:
        print(f"  ⚠️  Could not download {label}: {e}"); return False

    try:
        state = torch.load(local_path, map_location='cpu', weights_only=True)
    except Exception:
        state = torch.load(local_path, map_location='cpu')

    current = model.state_dict(); filtered = {}; skipped = []
    for k, v in state.items():
        if k not in current:
            skipped.append(f"  [not in model] {k}"); continue
        ms, cs = current[k].shape, v.shape
        if ms == cs:
            filtered[k] = v
        elif k == 'pos_emb.weight' and ms[1] == cs[1]:
            filtered[k] = F.interpolate(
                v.T.unsqueeze(0), size=ms[0], mode='linear', align_corners=True
            ).squeeze(0).T
            print(f"  📐 pos_emb interpolated {cs[0]}→{ms[0]}")
        else:
            skipped.append(f"  [shape {list(cs)}≠{list(ms)}] {k}")

    missing, _ = model.load_state_dict(filtered, strict=False)
    print(f"  ✅ {label}: loaded {len(filtered)} | random-init {len(missing)} | skipped {len(skipped)}")
    if skipped:
        for s in skipped[:5]: print(s)
        if len(skipped) > 5: print(f"  ... and {len(skipped)-5} more")
    del state, filtered; gc.collect()
    return True

load_checkpoint(model, HF_REPO, CHECKPOINT_SUB, HF_TOKEN, CHECKPOINT_SUB)
if torch.cuda.is_available(): torch.cuda.empty_cache()

# ════════════════════════════════════════════════════════════
#  Tokenizer & Conversational Dataset
# ════════════════════════════════════════════════════════════

print("\n─── Tokenizer & Dataset ────────────────────────────────")
tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token
EOS = tokenizer.eos_token_id

def get_dataset_ids():
    hf_cache_dir = os.path.expanduser("~/.cache/huggingface/datasets")
    for lock_file in glob.glob(f"{hf_cache_dir}/**/*.lock", recursive=True):
        try: os.remove(lock_file)
        except: pass

    all_ids = []
    texts = []

    try:
        print("Fetching DailyDialog (MTEB Mirror) ...")
        raw_dd = load_dataset("mteb/daily_dialog", "default")
        for split in ('train', 'validation'):
            for ex in raw_dd[split]:
                if isinstance(ex.get('text'), list) and len(ex['text']) > 1:
                    texts.append("\n".join([t.strip() for t in ex['text'] if t.strip()]))
        print(f"  ✓ Added DailyDialog ({len(texts):,} conversations)")
    except Exception as e:
        print(f"  ⚠️ DailyDialog failed: {e}")

    try:
        print("Fetching SAMSum Chats ...")
        raw_sam = load_dataset("samsum", trust_remote_code=True)
        sam_count = 0
        for split in ('train', 'validation'):
            for ex in raw_sam[split]:
                if ex.get('dialogue'):
                    texts.append(ex['dialogue'].strip())
                    sam_count += 1
        print(f"  ✓ Added SAMSum ({sam_count:,} conversations)")
    except Exception as e:
        print(f"  ⚠️ SAMSum failed: {e}")

    if not texts:
        print("⚠️ Conversational datasets failed. Falling back to WikiText...")
        raw_wiki = load_dataset('Salesforce/wikitext', 'wikitext-103-raw-v1', token=HF_TOKEN)
        for split in ('train', 'validation'):
            texts.extend([ex['text'].strip() for ex in raw_wiki[split] if ex['text'].strip()])

    print(f"\nTokenizing {len(texts):,} text blocks ...")
    chunk_size = 50000
    for i in range(0, len(texts), chunk_size):
        batch = texts[i : i + chunk_size]
        encoded_batch = tokenizer(batch, add_special_tokens=False)['input_ids']
        for seq in encoded_batch:
            all_ids.extend(seq + [EOS])

    print(f"  ✓ Total Tokens: {len(all_ids):,}")
    return all_ids

all_ids = get_dataset_ids()

n_val     = max(MAX_SEQ_LEN * 8, len(all_ids) // 20)
val_ids   = all_ids[:n_val]
train_ids = all_ids[n_val:]
print(f"  Train: {len(train_ids):,} tok  |  Val: {len(val_ids):,} tok")

import torch.utils.data as tud

class ChunkDataset(tud.Dataset):
    def __init__(self, ids: List[int], seq_len: int):
        self.seq_len = seq_len
        self.n_chunks = len(ids) // seq_len
        self.data = torch.tensor(ids[:self.n_chunks * seq_len], dtype=torch.long)

    def __len__(self):
        return self.n_chunks

    def __getitem__(self, i):
        start = i * self.seq_len
        return {'input_ids': self.data[start : start + self.seq_len]}

train_loader = tud.DataLoader(ChunkDataset(train_ids, MAX_SEQ_LEN), batch_size=BATCH_SIZE,
                               shuffle=True,  num_workers=2, pin_memory=True, drop_last=True)
valid_loader = tud.DataLoader(ChunkDataset(val_ids, MAX_SEQ_LEN),   batch_size=BATCH_SIZE,
                               shuffle=False, num_workers=2, pin_memory=True, drop_last=True)
print(f"  Train batches: {len(train_loader):,}  |  Val batches: {len(valid_loader):,}")

# ════════════════════════════════════════════════════════════
#  Eval + diversity metrics
# ════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate(loader, max_batches=30):
    model.eval(); total, n = 0.0, 0
    for i, batch in enumerate(loader):
        if i >= max_batches: break
        ids = batch['input_ids'].to(Config.DEVICE, non_blocking=True)
        out = model(ids, labels=ids)
        if out.get('loss') is not None and torch.isfinite(out['loss']):
            total += out['loss'].item(); n += 1
    model.train(); return total / max(n, 1)

PROMPTS = [
    "Hey, what do you think about",
    "I was wondering if you could help me",
    "So the thing is,",
    "Can you explain why",
]

@torch.no_grad()
def score_diversity(max_batches=30, gen_len=60, temperature=0.95, top_k=50, top_p=0.92,
                    rep_penalty=1.35):
    model.eval(); all_tokens: List[int] = []; rep_count = total_count = 0
    for prompt in PROMPTS:
        ids = tokenizer.encode(prompt, return_tensors='pt').to(Config.DEVICE); prev = -1
        for _ in range(gen_len):
            if ids.shape[1] >= MAX_SEQ_LEN: break
            logits = model(ids)['logits'][:, -1, :].float() / temperature
            seen = ids[0].unique()
            logits[0, seen] = torch.where(logits[0, seen] > 0,
                logits[0, seen] / rep_penalty, logits[0, seen] * rep_penalty)
            if top_k > 0:
                kth = torch.topk(logits, top_k).values[:, -1, None]
                logits = logits.masked_fill(logits < kth, float('-inf'))
            sl, si = torch.sort(logits, descending=True)
            cp = torch.cumsum(F.softmax(sl, dim=-1), dim=-1)
            sl[cp - F.softmax(sl, dim=-1) > top_p] = float('-inf')
            logits = torch.zeros_like(logits).scatter_(1, si, sl)
            next_t = torch.multinomial(F.softmax(logits, dim=-1), 1)[0,0].item()
            all_tokens.append(next_t)
            if next_t == prev: rep_count += 1
            total_count += 1; prev = next_t
            ids = torch.cat([ids, torch.tensor([[next_t]], device=Config.DEVICE)], dim=1)
    bg = list(zip(all_tokens, all_tokens[1:]))
    d1 = len(set(all_tokens)) / max(len(all_tokens), 1)
    d2 = len(set(bg))         / max(len(bg), 1)
    rr = rep_count             / max(total_count, 1)
    model.train(); return {'distinct1': d1, 'distinct2': d2, 'rep_rate': rr}

@torch.no_grad()
def show_samples(tag="", temperature=0.95, top_k=50, top_p=0.92,
                 rep_penalty=1.35, gen_len=80):
    model.eval()
    print(f"\n── Samples {tag} ──────────────────────────────────────")
    for prompt in PROMPTS[:3]:
        ids = tokenizer.encode(prompt, return_tensors='pt').to(Config.DEVICE)
        for _ in range(gen_len):
            if ids.shape[1] >= MAX_SEQ_LEN: break
            logits = model(ids)['logits'][:, -1, :].float() / temperature
            seen = ids[0].unique()
            logits[0, seen] = torch.where(logits[0, seen] > 0,
                logits[0, seen] / rep_penalty, logits[0, seen] * rep_penalty)
            if top_k > 0:
                kth = torch.topk(logits, top_k).values[:, -1, None]
                logits = logits.masked_fill(logits < kth, float('-inf'))
            sl, si = torch.sort(logits, descending=True)
            cp = torch.cumsum(F.softmax(sl, dim=-1), dim=-1)
            sl[cp - F.softmax(sl, dim=-1) > top_p] = float('-inf')
            logits = torch.zeros_like(logits).scatter_(1, si, sl)
            next_t = torch.multinomial(F.softmax(logits, dim=-1), 1)
            if next_t.item() == EOS: break
            ids = torch.cat([ids, next_t], dim=1)
        text = tokenizer.decode(ids[0], skip_special_tokens=True)
        print(f"  [{prompt}]\n  {text[len(prompt):].strip()[:300]}\n")
    model.train()

# ════════════════════════════════════════════════════════════
#  Optimizer
# ════════════════════════════════════════════════════════════

def _param_ids(modules):
    s = set()
    for m in modules: s.update(id(p) for p in m.parameters())
    return s

# chunk_decoder.lm_head.weight IS token_emb.weight (tied).
# Keep it in only one param group to avoid optimizer ValueError.
_TIED_IDS = {id(model.token_emb.weight)}

def _group(name, lr, modules, owns_tied=False):
    ids = _param_ids(modules)
    params = [p for p in model.parameters()
              if id(p) in ids and p.requires_grad
              and (owns_tied or id(p) not in _TIED_IDS)]
    return {'params': params, 'lr': lr, 'name': name}

# BUG-4 FIX: chunk_encoder and top_blocks get their own groups with
# higher LRs so they learn at the same rate as chunk_decoder.
param_groups = [
    _group('chunk_decoder',      LR_CHUNK_DECODER, [model.chunk_decoder]),
    _group('chunk_encoder',      LR_CHUNK_ENC,     [model.chunk_encoder]),
    _group('top_blocks',         LR_TOP,           list(model.top)),
    _group('horizon_predictor',  LR_HP,            [model.horizon_predictor]),
    _group('cif',                LR_CIF,           [model.cif]),
    _group('diamond_cross_attn', LR_DIAMOND,       [model.diamond]),
    _group('lattice',            LR_BACKBONE,      [model.lattice]),
    _group('bottom_blocks',      LR_BACKBONE,      list(model.bottom)),
    _group('pos_emb',            LR_EMBED,         [model.pos_emb]),
    _group('token_emb+ln_f',     LR_EMBED,         [model.token_emb, model.ln_f], owns_tied=True),
]

optimizer = torch.optim.AdamW(param_groups, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.95), eps=1e-8)

def _lr_lambda(step):
    if step < WARMUP_STEPS: return step / max(1, WARMUP_STEPS)
    t = (step - WARMUP_STEPS) / max(1, TOTAL_STEPS - WARMUP_STEPS)
    return max(0.1, math.cos(math.pi * t) * 0.5 + 0.5)

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)
scaler    = torch.cuda.amp.GradScaler(enabled=USE_AMP, init_scale=256, growth_interval=200)

print(f"\n─── Optimizer ──────────────────────────────────────────")
for g in param_groups:
    n = sum(p.numel() for p in g['params']) / 1e6
    print(f"  {g['name']:<22}  LR={g['lr']:.1e}  {n:.1f}M trainable params")

# ════════════════════════════════════════════════════════════
#  HF Upload helper
# ════════════════════════════════════════════════════════════

api = HfApi(token=HF_TOKEN)
SAVE_DIR = '/content' if os.path.exists('/content') else '/tmp'

def upload_to_hf(local_dir, subfolder, msg):
    print(f"  ☁️  Uploading → {HF_REPO}/{subfolder} ...")
    try:
        for root, _, files in os.walk(local_dir):
            for fname in files:
                lp = os.path.join(root, fname)
                api.upload_file(path_or_fileobj=lp,
                                path_in_repo=f"{subfolder}/{os.path.relpath(lp, local_dir)}",
                                repo_id=HF_REPO, repo_type='model', token=HF_TOKEN,
                                commit_message=msg)
        print(f"  ✅ {subfolder}")
    except Exception as e:
        print(f"  ⚠️  Upload failed: {e}")

# ════════════════════════════════════════════════════════════
#  Baseline
# ════════════════════════════════════════════════════════════

print(f"\n─── Baseline ({CHECKPOINT_SUB}) ──────────────────────────")
baseline_val = evaluate(valid_loader, max_batches=50)
print(f"  val_loss={baseline_val:.4f}  ppl={math.exp(min(baseline_val,20)):.1f}")

print("\n📏 Pre-training diversity ...")
pre_div = score_diversity(max_batches=30)
print(f"   distinct-1: {pre_div['distinct1']:.4f}  distinct-2: {pre_div['distinct2']:.4f}  "
      f"rep-rate: {pre_div['rep_rate']:.4f}")
show_samples("BEFORE TRAINING")

# ════════════════════════════════════════════════════════════
#  Training loop
# ════════════════════════════════════════════════════════════

print(f"\n{'='*65}")
print(f"🚀 Speaking finetuning — {CHECKPOINT_SUB} → {SAVE_SUBFOLDER}")
print(f"   Steps: {TOTAL_STEPS}  Eval every: {EVAL_EVERY}  Save every: {SAVE_EVERY}")
print(f"   Batch: {BATCH_SIZE}×{GRAD_ACCUM}={BATCH_SIZE*GRAD_ACCUM}")
print('='*65)

model.train()
train_iter    = iter(train_loader)
running_loss  = running_count = 0.0
best_val_loss = float('inf')
val_loss      = baseline_val
nan_skips     = 0
ema_loss      = None
EMA_ALPHA     = 0.95
t_start       = time.time()
best_dir      = os.path.join(SAVE_DIR, 'speaking_best')
optimizer.zero_grad()

for step in range(1, TOTAL_STEPS + 1):
    try:
        batch = next(train_iter)
    except StopIteration:
        train_iter = iter(train_loader); batch = next(train_iter)

    ids = batch['input_ids'].to(Config.DEVICE, non_blocking=True)

    with torch.amp.autocast('cuda', enabled=USE_AMP):
        out  = model(ids, labels=ids)
        loss = out['loss']
        if out.get('hp_loss') is not None and torch.isfinite(out['hp_loss']):
            loss = loss + HP_LOSS_WEIGHT * out['hp_loss']
        if loss > 50.0:
            loss = 50.0 + torch.log1p(loss - 50.0)
        loss = loss / GRAD_ACCUM

    if not torch.isfinite(loss):
        nan_skips += 1; optimizer.zero_grad()
        if nan_skips % 5 == 1:
            print(f"  ⚠️  step {step}: NaN loss (total={nan_skips})")
        if nan_skips > 10:
            for g in optimizer.param_groups: g['lr'] *= 0.5
            nan_skips = 0; print("  🔄 LRs halved due to persistent NaN")
        continue

    scaler.scale(loss).backward()
    running_loss  += loss.item() * GRAD_ACCUM
    running_count += 1

    if step % GRAD_ACCUM == 0:
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad and p.grad is not None],
            MAX_GRAD_NORM)
        prev_scale = scaler.get_scale()
        scaler.step(optimizer); scaler.update(); optimizer.zero_grad()

        if not torch.isfinite(grad_norm):
            nan_skips += 1
            if nan_skips % 5 == 1:
                print(f"  ⚠️  step {step}: NaN grads (scale {prev_scale:.0f}→{scaler.get_scale():.0f})")
        else:
            nan_skips = 0; scheduler.step()

    if step % 20 == 0 and running_count > 0:
        avg_loss = running_loss / running_count
        ema_loss = avg_loss if ema_loss is None else EMA_ALPHA * ema_loss + (1-EMA_ALPHA) * avg_loss
        running_loss = running_count = 0
        tok_s = step * BATCH_SIZE * MAX_SEQ_LEN / max(1, time.time() - t_start)
        ppl   = math.exp(min(avg_loss, 15)) if avg_loss < 50 else float('inf')
        print(f"  step {step:>5}/{TOTAL_STEPS}  loss={avg_loss:.4f}  ema={ema_loss:.4f}"
              f"  ppl={ppl:.1f}  lr={scheduler.get_last_lr()[0]:.1e}  "
              f"{tok_s:.0f} tok/s  (nan_skips={nan_skips})", flush=True)

    if step % EVAL_EVERY == 0:
        val_loss = evaluate(valid_loader, max_batches=30)
        ppl      = math.exp(min(val_loss, 20))
        improved = val_loss < best_val_loss
        print(f"\n  ✅ Step {step}  val={val_loss:.4f}  ppl={ppl:.1f}"
              + (" 💾 new best" if improved else ""))
        # Gradient norm per group — verifies hierarchical path is learning.
        # chunk_encoder and top_blocks should be non-zero after BUG-4 fix.
        gnorms = {}
        for g in optimizer.param_groups:
            norms = [p.grad.norm().item() for p in g['params']
                     if p.grad is not None and torch.isfinite(p.grad.norm())]
            gnorms[g['name']] = sum(norms) / max(len(norms), 1) if norms else 0.0
        key_groups = ['chunk_decoder', 'chunk_encoder', 'top_blocks']
        gstr = '  '.join(f"{k}={gnorms.get(k,0):.3f}" for k in key_groups)
        print(f"  ∇  {gstr}")
        show_samples(f"step {step}")

        if improved:
            best_val_loss = val_loss
            os.makedirs(best_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(best_dir, 'model.pt'))
            tokenizer.save_pretrained(best_dir)

        if step % SAVE_EVERY == 0:
            ckpt = os.path.join(SAVE_DIR, f'speaking_{step}')
            os.makedirs(ckpt, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(ckpt, 'model.pt'))
            tokenizer.save_pretrained(ckpt)
            upload_to_hf(ckpt, f"{SAVE_SUBFOLDER}/step_{step}",
                         f"Speaking finetune step {step} — val={val_loss:.4f}")
            shutil.rmtree(ckpt)

print(f"\n✅ Training done!  Best val: {best_val_loss:.4f}  "
      f"ppl={math.exp(min(best_val_loss,20)):.1f}  nan_skips={nan_skips}")

# ════════════════════════════════════════════════════════════
#  Final — load best, eval, scorecard, upload
# ════════════════════════════════════════════════════════════

if os.path.exists(os.path.join(best_dir, 'model.pt')):
    st = torch.load(os.path.join(best_dir, 'model.pt'),
                    map_location=Config.DEVICE, weights_only=True)
    model.load_state_dict(st, strict=False); print("✅ Best checkpoint loaded")

model.eval()
final_val = evaluate(valid_loader, max_batches=200)
print(f"Final val: {final_val:.4f}  ppl={math.exp(min(final_val,20)):.1f}")

print("\n📏 Post-training diversity ...")
post_div = score_diversity(max_batches=30)

show_samples("FINAL")

print("\n" + "="*65)
print("  SPEAKING SCORECARD")
print(f"  {'Metric':<16} {'Before':>10} {'After':>10}  Direction")
print(f"  {'-'*16} {'-'*10} {'-'*10}  {'-'*22}")
_metrics = [
    ("val loss",   f"{baseline_val:.4f}",            f"{final_val:.4f}",              "↓ lower=better"),
    ("perplexity", f"{math.exp(min(baseline_val,20)):.1f}", f"{math.exp(min(final_val,20)):.1f}", "↓ lower=better"),
    ("distinct-1", f"{pre_div['distinct1']:.4f}",    f"{post_div['distinct1']:.4f}",  "↑ higher=more diverse"),
    ("distinct-2", f"{pre_div['distinct2']:.4f}",    f"{post_div['distinct2']:.4f}",  "↑ higher=more diverse"),
    ("rep-rate",   f"{pre_div['rep_rate']:.4f}",     f"{post_div['rep_rate']:.4f}",   "↓ lower=less repetitive"),
]
for name, b, a, direction in _metrics:
    try:
        arrow = "✅" if (float(a) < float(b)) == ("↓" in direction) else "❌"
    except: arrow = ""
    print(f"  {name:<16} {b:>10} {a:>10}  {arrow} {direction}")
print("="*65)

upload_to_hf(best_dir, f"{SAVE_SUBFOLDER}/best",
             f"Speaking finetune best — val={final_val:.4f}")
print(f"\n🎉 Done! → thagnitti/io/{SAVE_SUBFOLDER}/best/")
