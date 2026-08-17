import torch
from io_v1 import Io, IoConfig

def generate_text():
    device = torch.device("cpu")
    print(f"Generating on device: {device}")
    
    # Use max_seq_len > 128 so we don't go out of bounds during generation
    cfg = IoConfig(
        vocab_size=256,
        d_model=256,
        n_heads=8,
        n_layers=6,
        max_seq_len=2048,  # Increased for generation
    )
    
    model = Io(cfg).to(device)
    
    # Load the trained model
    state_dict = torch.load("io_muon_model.pt", map_location=device, weights_only=True)
    # Ignore buffers that might have different sizes due to max_seq_len
    if 'pos_enc.abs_pe' in state_dict:
        del state_dict['pos_enc.abs_pe']
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    print("\nGenerating text...")
    prompt = "Once upon a time"
    # Convert string to list of ASCII values
    ids = torch.tensor([[ord(c) for c in prompt]], dtype=torch.long, device=device)
    
    with torch.no_grad():
        out_ids, stats = model.generate(ids, max_new=100, temperature=0.8, top_k=50)
        
    gen_text = "".join(chr(i) for i in out_ids[0].tolist() if i < 256)
    print("--------------------------------------------------")
    print(gen_text)
    print("--------------------------------------------------")
    print(f"Tokens: {stats['tokens']} | Accept rate: {stats['accept_rate']:.2f}")

if __name__ == "__main__":
    generate_text()
