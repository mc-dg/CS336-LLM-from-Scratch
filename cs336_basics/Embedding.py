import torch
import torch.nn as nn
import math 
class Embedding(nn.Module):
    
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        std = math.sqrt(2/(self.num_embeddings+self.embedding_dim))
        self.embedding_matrix = nn.Parameter(nn.init.trunc_normal_(torch.empty(self.num_embeddings, self.embedding_dim, device=device, dtype=dtype), std=std, a=-3*std, b=3*std))
        self.device = device
        self.dtype = dtype
            
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding_matrix[token_ids]
