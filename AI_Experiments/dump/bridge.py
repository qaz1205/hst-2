import sys
import torch
import json
import time
import tiktoken
from configuration_io import IoConfig
from modeling_io import IoForCausalLM
from trainer_utils import AsynchronousHFStreamer, setup_muon_adamw
from loss_guide import LossGuide

device = "cuda" if torch.cuda.is_available() else "cpu"
cfg = IoConfig(
    vocab_size=50257,
    d_model=256,
    n_heads=4,
    n_layers=6, 
    max_seq_len=256,
    dropout=0.1,
    lattice_depth=64,
    max_horizon=8,
)
enc = tiktoken.get_encoding("gpt2")
model = IoForCausalLM(cfg).to(device)

# Load existing model if it exists
try:
    import os
    if os.path.exists("model.pt"):
        model.load_state_dict(torch.load("model.pt", map_location=device))
        # print("Loaded existing model weights.")
except:
    pass

def train(max_steps, batch_size):
    max_steps = int(max_steps)
    batch_size = int(batch_size)
    print(f"🚀 Starting Training: {max_steps} steps, batch size {batch_size}")
    
    streamer = None
    try:
        streamer = AsynchronousHFStreamer(
            "flytech/python-codes-25k", 
            batch_size, 
            cfg.max_seq_len, 
            lambda t: enc.encode(t, allowed_special="all"),
            is_code=True
        )
        
        opt = setup_muon_adamw(model, 8e-4)
        
        # Initialize the Absolute Miracle
        guide = LossGuide(optimizer=opt, max_steps=max_steps, base_lr=12e-4, warmup_steps=int(max_steps * 0.1))
        
        model.train()
        
        t0 = time.time()
        for step in range(1, max_steps + 1):
            x, y = streamer.get_batch()
            x, y = x.to(device), y.to(device)
            
            out = model(input_ids=x, labels=y)
            loss = out.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            # Use Loss Guide to drive down the loss and compute dynamic LR
            current_lr = guide.step(step, loss.item())
            
            opt.step()
            opt.zero_grad(set_to_none=True)
            
            if step % 10 == 0:
                dt = time.time() - t0
                print(f"Step {step}/{max_steps} | Loss: {loss.item():.4f} | LR: {current_lr:.6f} | dt: {dt:.2f}s", flush=True)
                t0 = time.time()
        
        # Save model
        torch.save(model.state_dict(), "model.pt")
        print("✅ Training Complete and model saved to model.pt")
        
    finally:
        if streamer:
            streamer.stop_signal.set()

def chat(history_json, max_tokens, temp):
    history = json.loads(history_json)
    max_tokens = int(max_tokens)
    temp = float(temp)
    
    model.eval()
    prompt = ""
    for user, ai in history:
        if user: prompt += f"### Instruction:\n{user}\n"
        if ai: prompt += f"### Response:\n{ai}\n"
    
    header = prompt + "### Response:\n"
    ids = torch.tensor([enc.encode(header)], dtype=torch.long, device=device)
    output_tokens = model.generate_draft(ids, max_new_tokens=max_tokens, temperature=temp)
    raw_output = enc.decode(output_tokens.tolist()[0])
    new_text = raw_output[len(header):]
    print(new_text)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(1)
        
    cmd = sys.argv[1]
    if cmd == "train":
        train(sys.argv[2], sys.argv[3])
    elif cmd == "chat":
        chat(sys.argv[2], sys.argv[3], sys.argv[4])
