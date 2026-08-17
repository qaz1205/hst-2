import torch
import tiktoken
from io_v1 import Io, IoConfig

def generate_text():
    device = torch.device("cpu")
    print(f"Generating on device: {device}")
    
    enc = tiktoken.get_encoding("gpt2")
    vocab_size = enc.n_vocab
    
    # Needs to match the full model config
    cfg = IoConfig(
        vocab_size=vocab_size,
        d_model=768,
        n_heads=12,
        n_layers=12,
        max_seq_len=2048,  # Increased for generation
    )
    
    model = Io(cfg).to(device)
    
    # Load the trained model
    try:
        state_dict = torch.load("io_full_muon_model.pt", map_location=device, weights_only=True)
        # Ignore buffers that might have different sizes due to max_seq_len
        if 'pos_enc.abs_pe' in state_dict:
            del state_dict['pos_enc.abs_pe']
        model.load_state_dict(state_dict, strict=False)
        print("Successfully loaded trained weights.")
    except Exception as e:
        print(f"Could not load weights, generating with untrained model: {e}")
        
    model.eval()
    
    print("\nGenerating text...")
    prompt = "Once upon a time"
    # Encode prompt using tiktoken
    tokens = enc.encode(prompt)
    ids = torch.tensor([tokens], dtype=torch.long, device=device)
    
    with torch.no_grad():
        out_ids, stats = model.generate(ids, max_new=100, temperature=0.8, top_k=50)
        
    # Decode tokens back to string
    gen_text = enc.decode(out_ids[0].tolist())
    print("--------------------------------------------------")
    print(gen_text)
    print("--------------------------------------------------")
    print(f"Tokens: {stats['tokens']} | Accept rate: {stats['accept_rate']:.2f}")

if __name__ == "__main__":
    generate_text()
