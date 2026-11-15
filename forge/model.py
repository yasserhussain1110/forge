import torch
from torch import nn

from forge.decoder import DecoderLayer
from forge.rmsnorm import RMSNorm
from forge.rope import apply_llama3_scaling, compute_rope_inv_freq

class Transformer(nn.Module):
    inv_freq: torch.Tensor

    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            DecoderLayer(
                hidden_size=config.hidden_size,
                num_attention_heads=config.num_attention_heads,
                num_key_value_heads=config.num_key_value_heads,
                head_dim=config.head_dim,
                intermediate_size=config.intermediate_size,
                rms_norm_eps=config.rms_norm_eps
            )
            for _ in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        rope = config.rope_parameters
        inv_freq = compute_rope_inv_freq(rope["rope_theta"], config.head_dim)
        inv_freq = apply_llama3_scaling(
            inv_freq,
            factor=rope["factor"],
            low_freq_factor=rope["low_freq_factor"],
            high_freq_factor=rope["high_freq_factor"],
            original_max_position_embeddings=rope["original_max_position_embeddings"]
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, input_ids, attention_mask=None):
        hidden_states = self.embed_tokens(input_ids)
        batch, seq_len, _ = hidden_states.shape
        position_ids = torch.arange(seq_len, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(batch, seq_len)
        freqs = position_ids.unsqueeze(-1).float() * self.inv_freq
        emb = torch.cat((freqs, freqs), dim=-1)
        cos, sin = emb.cos(), emb.sin()
        position_embeddings = (cos, sin)
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=input_ids.device),
            diagonal=1
        )
        causal_mask = causal_mask[None, None, :, :]
        if attention_mask is None:
            attention_mask = causal_mask
        for layer in self.layers:
            hidden_states = layer(hidden_states, position_embeddings, attention_mask)
        hidden_states = self.norm(hidden_states)
        return hidden_states

class ForgeForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = Transformer(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, input_ids, attention_mask=None):
        hidden_states = self.model(input_ids, attention_mask=attention_mask)
        logits = self.lm_head(hidden_states)
        return logits