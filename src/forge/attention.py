import torch, math
from torch import nn
from forge.RoPE import do_rope

class Attention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, num_key_value_heads, head_dim, rope_theta):
        super().__init__()
        self.q_proj = nn.Linear(hidden_size, num_attention_heads * head_dim, False)
        self.k_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, False)
        self.v_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, False)
        self.o_proj = nn.Linear(num_attention_heads * head_dim, hidden_size, False)
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_head_dim = head_dim
        self.rope_theta = rope_theta

    def get_q_k_v(self, x, start):
        q_p = self.q_proj(x)
        k_p = self.k_proj(x)
        v_p = self.v_proj(x)
        q_p = q_p.view(-1, self.num_attention_heads, self.num_head_dim)
        k_p = k_p.view(-1, self.num_key_value_heads, self.num_head_dim)
        v_p = v_p.view(-1, self.num_key_value_heads, self.num_head_dim)
        q_rot = do_rope(q_p, start, self.rope_theta, self.num_head_dim)
        k_rot = do_rope(k_p, start, self.rope_theta, self.num_head_dim)
        q_rot = q_rot.movedim(0, 1)
        k_rot = k_rot.movedim(0, 1)
        v_p = v_p.movedim(0, 1)
        return (q_rot, k_rot, v_p)

    def forward(self, x, start, k_cache, v_cache):
        len = x.size()[0]
        q, k, v = self.get_q_k_v(x, start)
        k_cache[:,start:start+len,:] = k
        v_cache[:,start:start+len,:] = v
        repeats = self.num_attention_heads // self.num_key_value_heads
        v_repeated = torch.repeat_interleave(v_cache[:,:start+len,:], repeats=repeats, dim=0)
        k_repeated = torch.repeat_interleave(k_cache[:,:start+len,:], repeats=repeats, dim=0)
        k_swapped = k_repeated.movedim(1, 2)
        scores = q @ k_swapped / math.sqrt(self.num_head_dim)
        if scores.size()[1] > 1:
            mask = torch.triu(torch.ones_like(scores, dtype=torch.bool), diagonal=1)
            scores[mask] = -math.inf
        weights = torch.softmax(scores, dim=-1)
        out = weights @ v_repeated
        out = out.movedim(0, 1).reshape(-1, self.num_attention_heads * self.num_head_dim)
        return self.o_proj(out)
