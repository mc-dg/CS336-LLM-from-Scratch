import torch
import torch.nn as nn

from cs336_basics.Embedding import Embedding
from cs336_basics.Linear import Linear
from cs336_basics.RMSNorm import RMSNorm
from cs336_basics.transformer_block import transformer_block


class transformer_lm(nn.Module):
    def __init__(
        self,
        d_model,
        num_heads,
        vocab_size,
        context_length,
        num_layers,
        d_ff,
        theta=None,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.context_length = context_length
        self.theta = theta
        self.device = device
        self.dtype = dtype

        self.embedding = Embedding(vocab_size, d_model, self.device, self.dtype)
        self.layers = []
        for i in range(self.num_layers):
            block = transformer_block(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                max_seq_len=context_length,
                theta=theta,
                device=device,
                dtype=dtype,
            )
            self.add_module(f"layers{i}", block)
            self.layers.append(block)

        self.norm = RMSNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size, self.device, self.dtype)

    def forward(self, x, token_positions=None):
        x = self.embedding(x)

        T = x.shape[1]
        token_positions = torch.arange(T, device=x.device)

        for block in self.layers:
            x = block(x, token_positions=token_positions)

        x = self.norm(x)

        # 4. Linear (Output Embedding)
        logits = self.lm_head(x)

        return logits
