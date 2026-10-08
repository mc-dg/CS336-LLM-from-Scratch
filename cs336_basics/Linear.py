import torch
import torch.nn as nn
import math 
class Linear(nn.Module):
    
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        std = math.sqrt(2/(self.in_features+self.out_features))
        self.weight = nn.Parameter(nn.init.trunc_normal_(torch.empty(self.out_features, self.in_features, device=device, dtype=dtype), std=std, a=-3*std, b=3*std))
        self.device = device
        self.dtype = dtype
        
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x@self.weight.t()
