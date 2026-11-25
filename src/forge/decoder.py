from torch import nn
from forge.rmsnorm import RMSNorm
from forge.attention import Attention
from forge.mlp import MLP

class Decoder(nn.Module):
    def __init__(self, hidden_size, intermediate_size, num_attention_heads, num_key_value_heads, head_dim, rope_theta, rms_norm_eps):
        super().__init__()
        self.input_layernorm = RMSNorm(hidden_size, rms_norm_eps)
        self.self_attn = Attention(hidden_size, num_attention_heads, num_key_value_heads, head_dim, rope_theta)
        self.post_attention_layernorm = RMSNorm(hidden_size, rms_norm_eps)
        self.mlp = MLP(hidden_size, intermediate_size)

    def forward(self, x):
        x = x + self.self_attn(self.input_layernorm(x))
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x