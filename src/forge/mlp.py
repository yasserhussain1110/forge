from torch import nn
from torch.nn import functional as F

class MLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, False)

    def forward(self, x):
        g = F.silu(self.gate_proj(x))
        u = self.up_proj(x)
        m = g * u
        return self.down_proj(m)