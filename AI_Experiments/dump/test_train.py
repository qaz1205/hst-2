import torch
import time
from configuration_io import IoConfig
from modeling_io import IoForCausalLM
from trainer_utils import setup_muon_adamw

print("Creating model...", flush=True)
cfg = IoConfig(vocab_size=50257, d_model=256, n_heads=4, n_layers=6, max_seq_len=256, dropout=0.1, lattice_depth=64, max_horizon=8)
model = IoForCausalLM(cfg)
print(f"Model params: {sum(p.numel() for p in model.parameters()):,}", flush=True)

opt = setup_muon_adamw(model, 8e-4)
model.train()
x = torch.randint(0, 50257, (1, 64))

for step in range(1, 4):
    t0 = time.time()
    out = model(input_ids=x, use_cache=False, labels=x)
    loss = out.loss
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    t1 = time.time()
    print(f"Step {step}: Loss={loss.item():.4f} Fwd+Bwd={t1-t0:.2f}s", flush=True)
    opt.step()
    opt.zero_grad(set_to_none=True)
    t2 = time.time()
    print(f"  Opt step={t2-t1:.2f}s", flush=True)

print("Training loop OK!", flush=True)
