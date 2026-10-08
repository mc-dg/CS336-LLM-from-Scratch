import torch
import torch.nn as nn
from cs336_basics.RMSNorm import RMSNorm
from cs336_basics.multihead_self_attention import multi_head_self_attention
from cs336_basics.SwiGLU import SwiGLU

class transformer_block(nn.Module):
        
    def __init__(self, d_model, num_heads, d_ff, max_seq_len=None, theta=None, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.device = device
        self.dtype = dtype
        
        self.mhsa = multi_head_self_attention(
            d_model, num_heads, max_seq_len=max_seq_len, theta=theta, device=device, dtype=dtype
        )
        self.ln1 = RMSNorm(d_model)
        self.ln2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff)

    def forward(self, x, token_positions=None):
        att = x + self.mhsa(self.ln1(x), token_positions=token_positions)
        out = att + self.ffn(self.ln2(att))
        return out