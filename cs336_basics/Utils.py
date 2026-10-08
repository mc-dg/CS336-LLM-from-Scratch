import torch
from torch import Tensor
from jaxtyping import Bool, Float, Int
import math


def soft_max(x, i) ->torch.Tensor:
    max_val = torch.max(x, dim=i, keepdim=True).values
    exp_x= torch.exp(x-max_val)
    denom= torch.sum(exp_x, dim=i, keepdim=True)
    return exp_x/denom

def scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... keys d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    d_k = Q.shape[-1]
    scores = (Q @ K.transpose(-2, -1)) / math.sqrt(d_k)
    
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
        
    att_weights = soft_max(scores, -1)
    
    return att_weights @ V