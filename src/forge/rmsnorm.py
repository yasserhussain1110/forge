import torch.nn as nn
import torch

class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, tensor):
        sq = tensor * tensor
        sq_mean = sq.mean(dim=-1, keepdim=True)
        rev_sqrt = torch.rsqrt(sq_mean + self.eps)
        return tensor * rev_sqrt * self.weight