from transformers import PretrainedConfig

class IoConfig(PretrainedConfig):
    model_type = "io"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=50257,
        d_model=256,
        n_heads=4,
        n_layers=4,
        max_seq_len=256,
        dropout=0.1,
        lattice_depth=64,
        max_horizon=8,
        fb_iterations=1,
        hebbian_decay=0.995,
        exit_threshold=0.90,
        bos_token_id=50256,
        eos_token_id=50256,
        pad_token_id=50256,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        self.lattice_depth = lattice_depth
        self.max_horizon = max_horizon
        self.fb_iterations = fb_iterations
        self.hebbian_decay = hebbian_decay
        self.exit_threshold = exit_threshold
        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            **kwargs,
        )
