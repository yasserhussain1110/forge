import torch
from torch.testing import assert_close
from transformers.models.llama.modeling_llama import LlamaAttention

from forge.attention import Attention
from forge.rope import compute_rope_inv_freq

class TinyConfig:
    hidden_size = 16
    num_attention_heads = 4
    num_key_value_heads = 2
    head_dim = 4
    attention_bias = False
    attention_dropout = 0.0
    _attn_implementation = "eager"

def test_attention_matches_huggingface():
    torch.manual_seed(0)
    config = TinyConfig()
    reference = LlamaAttention(config, layer_idx=0)  # pyright: ignore[reportArgumentType]
    mine = Attention(
        hidden_size=config.hidden_size,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads,
        head_dim=config.head_dim
    )
    mine.load_state_dict(reference.state_dict())
    batch, seq_len = 2, 4
    x = torch.randn(batch, seq_len, config.hidden_size)

    position_ids = torch.arange(seq_len).unsqueeze(0).repeat(batch, 1)
    inv_freq = compute_rope_inv_freq(10000.0, config.head_dim)
    freqs = position_ids.unsqueeze(-1).float() * inv_freq
    emb = torch.cat((freqs, freqs), dim=-1)
    cos, sin = emb.cos(), emb.sin()
    causal_mask = torch.triu(
        torch.full((seq_len, seq_len), float("-inf")),
        diagonal=1
    )
    attention_mask = causal_mask[None, None, :, :]
    with torch.no_grad():
        mine_out = mine(x, (cos, sin), attention_mask)
        ref_out, _ = reference(
            x,
            position_embeddings=(cos, sin),
            attention_mask=attention_mask
        )
        assert_close(mine_out, ref_out)