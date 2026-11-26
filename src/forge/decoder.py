from torch import nn, zeros
from forge.rmsnorm import RMSNorm
from forge.attention import Attention
from forge.mlp import MLP

class Decoder(nn.Module):
    def __init__(self, hidden_size, intermediate_size, num_attention_heads, num_key_value_heads, head_dim, rope_theta, rms_norm_eps, max_sequence_length):
        super().__init__()
        self.input_layernorm = RMSNorm(hidden_size, rms_norm_eps)
        self.self_attn = Attention(hidden_size, num_attention_heads, num_key_value_heads, head_dim, rope_theta)
        self.post_attention_layernorm = RMSNorm(hidden_size, rms_norm_eps)
        self.mlp = MLP(hidden_size, intermediate_size)
        self.num_key_value_heads = num_key_value_heads
        self.max_sequence_length = max_sequence_length
        self.head_dim = head_dim
        self.register_buffer("k", zeros(self.num_key_value_heads, self.max_sequence_length, self.head_dim), persistent=False)
        self.register_buffer("v", zeros(self.num_key_value_heads, self.max_sequence_length, self.head_dim), persistent=False)
        self.reset_kv_cache()

    def forward(self, x):
        x = x + self.self_attn(self.input_layernorm(x), self.cached_filled_size, self.k, self.v)
        x = x + self.mlp(self.post_attention_layernorm(x))
        self.cached_filled_size += x.size()[0]
        return x

    def reset_kv_cache(self):
        self.cached_filled_size = 0