import torch
from torch import nn
from forge.rope import apply_rotary_pos_emb

def repeat_kv(x, n_rep):
    batch, num_kv_heads, seq_len, head_dim = x.shape
    if n_rep == 1:
        return x
    x = x[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, seq_len, head_dim)
    return x.reshape(batch, num_kv_heads * n_rep, seq_len, head_dim)

class Attention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, num_key_value_heads, head_dim):
        super().__init__()
        self.head_dim = head_dim
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_key_value_groups = num_attention_heads // num_key_value_heads
        self.scaling = head_dim ** -0.5

        self.q_proj = nn.Linear(hidden_size, num_attention_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_attention_heads * head_dim, hidden_size, bias=False)

    def forward(self, hidden_states, position_embeddings, attention_mask=None):
        batch, seq_len, _ = hidden_states.shape
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)
        query = query.view(batch, seq_len, self.num_attention_heads, self.head_dim).transpose(1,2)
        key = key.view(batch, seq_len, self.num_key_value_heads, self.head_dim).transpose(1,2)
        value = value.view(batch, seq_len, self.num_key_value_heads, self.head_dim).transpose(1,2)
        cos, sin = position_embeddings
        query, key = apply_rotary_pos_emb(query, key, cos, sin)
        key = repeat_kv(key, self.num_key_value_groups)
        value = repeat_kv(value, self.num_key_value_groups)
        scores = torch.matmul(query, key.transpose(2, 3)) * self.scaling
        if attention_mask is not None:
            scores = scores + attention_mask
        weights = torch.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
        attn_output = torch.matmul(weights, value)
        attn_output = attn_output.transpose(1,2).contiguous()
        attn_output = attn_output.reshape(batch, seq_len, -1)
        return self.o_proj(attn_output)
    