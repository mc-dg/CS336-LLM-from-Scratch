import math

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype
        std = math.sqrt(1 / self.d_model)
        self.weight = nn.Parameter(
            nn.init.trunc_normal_(
                torch.empty(self.d_model, device=self.device, dtype=torch.float32),
                std=std,
                a=-3 * std,
                b=3 * std,
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        mean = torch.mean(x**2, dim=-1, keepdim=True)
        denom = torch.sqrt(mean + self.eps)
        before_g = x / denom
        result = before_g * self.weight
        return result.to(in_dtype)
