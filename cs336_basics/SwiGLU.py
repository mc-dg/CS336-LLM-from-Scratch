import math

import torch
import torch.nn as nn


class SwiGLU(nn.Module):
    def _get_d_ff(self, d_model, multiple_of=64) -> int:
        ideal_d_ff = int(2 * d_model * 4 / 3)
        d_ff = multiple_of * math.floor(ideal_d_ff / multiple_of)
        return d_ff

    def __init__(self, d_model, d_ff=None, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff if d_ff is not None else self._get_d_ff(d_model)
        std = math.sqrt(1 / (self.d_model + self.d_model))
        self.w1 = nn.Parameter(
            nn.init.trunc_normal_(
                torch.empty(self.d_ff, self.d_model, device=device, dtype=dtype),
                std=std,
                a=-3 * std,
                b=3 * std,
            )
        )
        self.w2 = nn.Parameter(
            nn.init.trunc_normal_(
                torch.empty(self.d_model, self.d_ff, device=device, dtype=dtype),
                std=std,
                a=-3 * std,
                b=3 * std,
            )
        )
        self.w3 = nn.Parameter(
            nn.init.trunc_normal_(
                torch.empty(self.d_ff, self.d_model, device=device, dtype=dtype),
                std=std,
                a=-3 * std,
                b=3 * std,
            )
        )

        self.device = device
        self.dtype = dtype

    def _swish(self, x, W):
        return (x @ W.t()) * torch.sigmoid(x @ W.t())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        swish = self._swish(x, self.w1)
        return (swish * (x @ self.w3.t())) @ self.w2.t()
