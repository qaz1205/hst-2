import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import time
import math
import sys

# Import Io from io_v1
from io_v1 import Io, IoConfig, compute_loss, CurriculumScheduler

# ---------------------------------------------------------------------------
# MuonAdamW Optimizer from Karpathy's autoresearch
# ---------------------------------------------------------------------------

polar_express_coeffs = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]

# Note: removing torch.compile to avoid issues on CPU/older torch versions without proper backend
def adamw_step_fused(p, grad, exp_avg, exp_avg_sq, step_t, lr_t, beta1_t, beta2_t, eps_t, wd_t):
    p.mul_(1 - lr_t * wd_t)
    exp_avg.lerp_(grad, 1 - beta1_t)
    exp_avg_sq.lerp_(grad.square(), 1 - beta2_t)
    bias1 = 1 - beta1_t ** step_t
    bias2 = 1 - beta2_t ** step_t
    denom = (exp_avg_sq / bias2).sqrt() + eps_t
    step_size = lr_t / bias1
    p.add_(exp_avg / denom, alpha=-step_size)

def muon_step_fused(stacked_grads, stacked_params, momentum_buffer, second_momentum_buffer,
                    momentum_t, lr_t, wd_t, beta2_t, ns_steps, red_dim):
    # Nesterov momentum
    momentum = momentum_t.to(stacked_grads.dtype)
    momentum_buffer.lerp_(stacked_grads, 1 - momentum)
    g = stacked_grads.lerp_(momentum_buffer, momentum)
    
    # Polar express orthogonalization
    X = g.bfloat16() if g.device.type != "cpu" else g.float()
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.02 + 1e-6)
    if g.size(-2) > g.size(-1):
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X.mT @ X
            B = b * A + c * (A @ A)
            X = a * X + X @ B
    else:
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X @ X.mT
            B = b * A + c * (A @ A)
            X = a * X + B @ X
    g = X.to(g.dtype)
    
    # NorMuon variance reduction
    beta2 = beta2_t.to(g.dtype)
    v_mean = g.float().square().mean(dim=red_dim, keepdim=True)
    red_dim_size = g.size(red_dim)
    v_norm_sq = v_mean.sum(dim=(-2, -1), keepdim=True) * red_dim_size
    v_norm = v_norm_sq.sqrt()
    second_momentum_buffer.lerp_(v_mean.to(dtype=second_momentum_buffer.dtype), 1 - beta2)
    step_size = second_momentum_buffer.clamp_min(1e-10).rsqrt()
    scaled_sq_sum = (v_mean * red_dim_size) * step_size.float().square()
    v_norm_new = scaled_sq_sum.sum(dim=(-2, -1), keepdim=True).sqrt()
    final_scale = step_size * (v_norm / v_norm_new.clamp_min(1e-10))
    g = g * final_scale.to(g.dtype)
    
    # Cautious weight decay + parameter update
    lr = lr_t.to(g.dtype)
    wd = wd_t.to(g.dtype)
    mask = (g * stacked_params) >= 0
    stacked_params.sub_(lr * g + lr * wd * stacked_params * mask)

class MuonAdamW(torch.optim.Optimizer):
    """Combined optimizer: Muon for 2D matrix params, AdamW for others."""

    def __init__(self, param_groups):
        super().__init__(param_groups, defaults={})
        self._adamw_step_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta1_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_eps_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_momentum_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")

    def _step_adamw(self, group):
        for p in group['params']:
            if p.grad is None:
                continue
            grad = p.grad
            state = self.state[p]
            if not state:
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_sq'] = torch.zeros_like(p)
            state['step'] += 1
            self._adamw_step_t.fill_(state['step'])
            self._adamw_lr_t.fill_(group['lr'])
            self._adamw_beta1_t.fill_(group['betas'][0])
            self._adamw_beta2_t.fill_(group['betas'][1])
            self._adamw_eps_t.fill_(group['eps'])
            self._adamw_wd_t.fill_(group['weight_decay'])
            adamw_step_fused(p, grad, state['exp_avg'], state['exp_avg_sq'],
                            self._adamw_step_t, self._adamw_lr_t, self._adamw_beta1_t,
                            self._adamw_beta2_t, self._adamw_eps_t, self._adamw_wd_t)

    def _step_muon(self, group):
        params = group['params']
        if not params:
            return
            
        # Ensure all params have gradients (fill None with zeros)
        for p in params:
            if p.grad is None:
                p.grad = torch.zeros_like(p)
                
        p = params[0]
        state = self.state[p]
        num_params = len(params)
        shape, device, dtype = p.shape, p.device, p.dtype
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros(num_params, *shape, dtype=dtype, device=device)
        if "second_momentum_buffer" not in state:
            state_shape = (num_params, shape[-2], 1) if shape[-2] >= shape[-1] else (num_params, 1, shape[-1])
            state["second_momentum_buffer"] = torch.zeros(state_shape, dtype=dtype, device=device)
        red_dim = -1 if shape[-2] >= shape[-1] else -2
        stacked_grads = torch.stack([p.grad for p in params])
        stacked_params = torch.stack(params)
        self._muon_momentum_t.fill_(group["momentum"])
        self._muon_beta2_t.fill_(group["beta2"] if group["beta2"] is not None else 0.0)
        self._muon_lr_t.fill_(group["lr"] * max(1.0, shape[-2] / shape[-1])**0.5)
        self._muon_wd_t.fill_(group["weight_decay"])
        muon_step_fused(stacked_grads, stacked_params,
                        state["momentum_buffer"], state["second_momentum_buffer"],
                        self._muon_momentum_t, self._muon_lr_t, self._muon_wd_t,
                        self._muon_beta2_t, group["ns_steps"], red_dim)
        torch._foreach_copy_(params, list(stacked_params.unbind(0)))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            if group['kind'] == 'adamw':
                self._step_adamw(group)
            elif group['kind'] == 'muon':
                self._step_muon(group)


def setup_muon_adamw(model, learning_rate):
    """Sort parameters into Muon (2D matrix) and AdamW (others) groups."""
    matrix_params = []
    other_params = []
    
    for name, p in model.named_parameters():
        if p.requires_grad:
            if len(p.shape) == 2:
                matrix_params.append(p)
            else:
                other_params.append(p)
                
    param_groups = [
        dict(kind='adamw', params=other_params, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.01)
    ]
    
    for shape in sorted({p.shape for p in matrix_params}):
        group_params = [p for p in matrix_params if p.shape == shape]
        param_groups.append(dict(
            kind='muon', params=group_params, lr=learning_rate * 5.0, # Muon generally handles higher LR
            momentum=0.95, ns_steps=5, beta2=0.95, weight_decay=0.01,
        ))
        
    optimizer = MuonAdamW(param_groups)
    for group in optimizer.param_groups:
        group["initial_lr"] = group["lr"]
    return optimizer


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    # 1. Config for < 2GB model (A 30M param model for CPU training speed)
    cfg = IoConfig(
        vocab_size=256,       # Byte-level tokenization for simplicity and speed
        d_model=256,
        n_heads=8,
        n_layers=6,
        max_seq_len=128,
        batch_size=8,
        max_steps=100,        # Fast proof-of-concept run
        lr=5e-4
    )
    
    # 2. Setup Data
    data_path = "../Development/HST v5 Training logs/hst_v5_chunk_story_100k_chars.txt"
    try:
        raw_text = open(data_path, encoding="utf-8").read()
        print(f"Loaded {len(raw_text)} chars of training data.")
    except Exception as e:
        print(f"Could not load data: {e}")
        return
        
    train_data = torch.tensor([ord(c) for c in raw_text if ord(c) < 256], dtype=torch.long)
    
    # 3. Model
    model = Io(cfg).to(device)
    print(f"Io Model Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M (Target: < 2 GB)")
    
    # 4. Use Autoresearch Muon optimizer
    opt = setup_muon_adamw(model, cfg.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg.max_steps)
    cur = CurriculumScheduler(cfg, model.cif)
    
    # 5. Training loop
    model.train()
    t0 = time.time()
    
    for step in range(1, cfg.max_steps + 1):
        ix = torch.randint(len(train_data) - cfg.max_seq_len - 1, (cfg.batch_size,))
        x = torch.stack([train_data[i:i+cfg.max_seq_len] for i in ix]).to(device)
        y = torch.stack([train_data[i+1:i+cfg.max_seq_len+1] for i in ix]).to(device)
        
        opt.zero_grad()
        out = model(x, training=True)
        loss = compute_loss(out, y, model.cif, cur.horizon())
        loss.backward()
        
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        cur.tick()
        
        if step % 20 == 0:
            dt = time.time() - t0
            print(f"Step {step:4d} | Loss: {loss.item():.4f} | dt: {dt:.2f}s")
            t0 = time.time()
            
    torch.save(model.state_dict(), "io_muon_model.pt")
    print("Training complete! Model saved to io_muon_model.pt")

    # 6. Generate Coherent Text
    print("\\nGenerating text...")
    model.eval()
    prompt = "Once upon a time"
    ids = torch.tensor([[ord(c) for c in prompt]], dtype=torch.long, device=device)
    with torch.no_grad():
        out_ids, stats = model.generate(ids, max_new=100, temperature=0.8, top_k=50)
        
    gen_text = "".join(chr(i) for i in out_ids[0].tolist())
    print("--------------------------------------------------")
    print(gen_text)
    print("--------------------------------------------------")

if __name__ == "__main__":
    train()
