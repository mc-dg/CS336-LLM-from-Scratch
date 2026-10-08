import torch
import torch.nn as nn


class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device

        # buffers
        positions = torch.arange(max_seq_len)[:, None]
        dims = torch.arange(0, self.d_k, 2)[None, :]
        angles = positions / (self.theta ** (dims / self.d_k))
        self.register_buffer("sin_precomputed", torch.sin(angles), persistent=False)
        self.register_buffer("cos_precomputed", torch.cos(angles), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        cos = self.cos_precomputed[token_positions]
        sin = self.sin_precomputed[token_positions]

        if token_positions.ndim == 1:
            cos = cos.unsqueeze(0)
            sin = sin.unsqueeze(0)

        x1 = x[..., 0::2]
        x2 = x[..., 1::2]

        out1 = x1 * cos - x2 * sin
        out2 = x1 * sin + x2 * cos

        return torch.stack([out1, out2], dim=-1).flatten(-2)
