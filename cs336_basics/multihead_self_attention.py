import math

import torch
import torch.nn as nn
from einops import rearrange

from cs336_basics.RoPE import RoPE
from cs336_basics.Utils import scaled_dot_product_attention


class multi_head_self_attention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int = None,
        theta: float = None,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = self.d_v = d_model // num_heads

        self.max_seq_len = max_seq_len
        self.theta = theta
        if theta is not None:
            self.rope = RoPE(self.theta, self.d_k, self.max_seq_len)

        std_qkv = math.sqrt(2 / (self.d_model + 3 * self.d_model))
        std_o = math.sqrt(2 / (self.d_model + self.d_model))

        self.W_qkv = nn.Parameter(
            nn.init.trunc_normal_(
                torch.empty(self.d_model, 3 * self.d_model, device=device, dtype=dtype),
                std=std_qkv,
                a=-3 * std_qkv,
                b=3 * std_qkv,
            )
        )
        self.W_o = nn.Parameter(
            nn.init.trunc_normal_(
                torch.empty(self.d_model, self.d_model, device=device, dtype=dtype),
                std=std_o,
                a=-3 * std_o,
                b=3 * std_o,
            )
        )

        self.device = device
        self.dtype = dtype

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor = None) -> torch.Tensor:
        B, T, C = x.shape
        qkv = x @ self.W_qkv

        q, k, v = rearrange(qkv, "b t (three h d_k) -> three b h t d_k", three=3, h=self.num_heads)

        causal_mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=0)

        if token_positions is not None:
            q, k = self.rope.forward(q, token_positions), self.rope.forward(k, token_positions)

        att = scaled_dot_product_attention(q, k, v, causal_mask)
        out_concatenated = rearrange(att, "b h t d_k -> b t (h d_k)")

        return out_concatenated @ self.W_o
