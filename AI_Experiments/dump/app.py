import gradio as gr
import torch
import time
import tiktoken
from configuration_io import IoConfig
from modeling_io import IoForCausalLM
from trainer_utils import AsynchronousHFStreamer, setup_muon_adamw

# Use deepseek-coder layout & paradigm (Trainer + Chat + FIM components)

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
# Use tiktoken encoding
enc = tiktoken.get_encoding("gpt2")
model = IoForCausalLM(cfg).to(device)

def get_lr(it, max_iters, lr_init=8e-4, warmup=100):
    if it < warmup: return lr_init * it / warmup
    if it > max_iters: return lr_init * 0.1
    import math
    decay_ratio = (it - warmup) / (max_iters - warmup)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return lr_init * 0.1 + coeff * (lr_init - lr_init * 0.1)

def train_model(max_steps, batch_size):
    streamer = None
    logs = [f"🚀 Starting Io-Coder Pre-training on {device.upper()}!"]
    try:
        yield "\n".join(logs)
        t0 = time.time()
        
        # Train on code! Like DeepSeek Coder does.
        logs.append(f"📦 Connecting to dataset: HuggingFaceH4/ultrachat_200k...")
        yield "\n".join(logs)
        
        try:
            streamer = AsynchronousHFStreamer(
                "HuggingFaceH4/ultrachat_200k", 
                int(batch_size), 
                cfg.max_seq_len, 
                lambda t: enc.encode(t, allowed_special="all"),
                is_code=False,
                split="train_sft"
            )
        except Exception as e:
            logs.append(f"❌ Data Error: {e}")
            yield "\n".join(logs)
            return

        opt = setup_muon_adamw(model, 8e-4)
        model.train()
        training_failed = False
        
        for step in range(1, int(max_steps) + 1):
            try:
                lr = get_lr(step, int(max_steps))
                for pg in opt.param_groups:
                    pg['lr'] = lr * 5.0 if pg.get('kind') == 'muon' else lr
                    
                x, y = streamer.get_batch()
                x, y = x.to(device), y.to(device)
                
                # Forward pass using HF paradigm
                out = model(input_ids=x, use_cache=False, labels=y)
                loss = out.loss
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                
                if step % 5 == 0:
                    dt = time.time() - t0
                    logs.append(f"Step {step:4d}/{int(max_steps)} | Loss: {loss.item():.4f} | dt: {dt:.2f}s")
                    if len(logs) > 20: logs = logs[-20:]
                    yield "\n".join(logs)
                    t0 = time.time()
            except Exception as e:
                logs.append(f"❌ Error at step {step}: {e}")
                yield "\n".join(logs)
                training_failed = True
                break
                
        model.eval()
        if not training_failed:
            logs.append("\n✅ Training Complete! Io is ready for generation.")
        yield "\n".join(logs)
    finally:
        if streamer:
            streamer.stop_signal.set()

def generate_chat(history, max_tokens, temp):
    model.eval()
    try:
        # Prompt formatting similar to deepseek-coder
        prompt = ""
        for user, ai in history:
            if user: prompt += f"### Instruction:\n{user}\n"
            if ai: prompt += f"### Response:\n{ai}\n"
            
        header = prompt + "### Response:\n"
        ids = torch.tensor([enc.encode(header)], dtype=torch.long, device=device)
        output_tokens = model.generate_draft(ids, max_new_tokens=int(max_tokens), temperature=temp)
        # Decode and extract just the new part
        raw_output = enc.decode(output_tokens)
        new_text = raw_output[len(header):]
        return new_text
    except Exception as e:
        return f"Error: {e}"

def chat_interface(user_message, history, max_tokens, temp):
    if not history:
        history = []
    history.append((user_message, None))
    response = generate_chat(history, max_tokens, temp)
    history[-1] = (user_message, response)
    return "", history

def evaluate_fim(prefix, suffix, max_tokens, temp):
    # FIM template: <|fim_prefix|> ... <|fim_suffix|> ... <|fim_middle|>
    model.eval()
    prompt = f"<|fim_prefix|>{prefix}<|fim_suffix|>{suffix}<|fim_middle|>"
    try:
        ids = torch.tensor([enc.encode(prompt)], dtype=torch.long, device=device)
        output_tokens = model.generate_draft(ids, max_new_tokens=int(max_tokens), temperature=temp)
        raw_output = enc.decode(output_tokens)
        middle_text = raw_output[len(prompt):]
        # Return merged view
        return f"{prefix}{middle_text}{suffix}"
    except Exception as e:
        return f"Error: {e}"

with gr.Blocks(theme=gr.themes.Monochrome()) as demo:
    gr.Markdown("# 🪐 Io Architecture \n**(UltraChat-200k · Pell-Lucas · CIF · HuggingFace)**")
    
    with gr.Tab("1. Pre-Train Io-Coder"):
        with gr.Row():
            max_s = gr.Slider(50, 10000, 200, step=10, label="Max Steps")
            batch_s = gr.Slider(1, 16, 4, step=1, label="Batch Size")
        log_out = gr.Code(label="Training Logs", lines=20)
        train_btn = gr.Button("Pre-Train on HuggingFaceH4/ultrachat_200k", variant="primary")
        train_btn.click(train_model, [max_s, batch_s], log_out)
        
    with gr.Tab("2. Io-Coder Chat"):
        chatbot = gr.Chatbot(height=400)
        msg = gr.Textbox(placeholder="E.g., Write a python function to compute fibonacci...", label="User Message")
        with gr.Row():
            c_max_t = gr.Slider(10, 200, 50, step=1, label="Max Tokens")
            c_temp = gr.Slider(0.1, 1.5, 0.85, label="Temperature")
        chat_btn = gr.Button("Send", variant="primary")
        chat_btn.click(chat_interface, [msg, chatbot, c_max_t, c_temp], [msg, chatbot])
        
    with gr.Tab("3. FIM (Fill-In-the-Middle)"):
        gr.Markdown("Test standard Code Completion via FIM tokens.")
        with gr.Row():
            pref = gr.Code(label="Prefix", lines=5, value="def quicksort(arr):\\n    if len(arr) <= 1:\\n        return arr")
            suff = gr.Code(label="Suffix", lines=5, value="    return quicksort(left) + middle + quicksort(right)")
        with gr.Row():
            f_max_t = gr.Slider(10, 200, 50, step=1, label="Max New Tokens")
            f_temp = gr.Slider(0.1, 1.5, 0.85, label="Temperature")
        fim_btn = gr.Button("Complete Middle", variant="primary")
        fim_out = gr.Code(label="Reconstructed Code")
        fim_btn.click(evaluate_fim, [pref, suff, f_max_t, f_temp], fim_out)
        
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 7860 if "SPACE_ID" in os.environ else 3000))
    demo.queue().launch(
        server_name="0.0.0.0", 
        server_port=port, 
        show_api=False,     # ← This helps a lot
        debug=True
    )
