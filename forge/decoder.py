from torch import nn

from forge.attention import Attention
from forge.mlp import MLP
from forge.rmsnorm import RMSNorm

class DecoderLayer(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, num_key_value_heads, head_dim, intermediate_size, rms_norm_eps):
        super().__init__()
        self.self_attn = Attention(
            hidden_size,
            num_attention_heads,
            num_key_value_heads,
            head_dim
        )
        self.mlp = MLP(hidden_size, intermediate_size)
        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)

    def forward(self, hidden_states, position_embeddings, attention_mask=None):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, position_embeddings, attention_mask)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states