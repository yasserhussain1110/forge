import torch
from torch.testing import assert_close
from transformers import AutoConfig
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

from forge.rope import apply_llama3_scaling, compute_rope_inv_freq

def test_rope_inv_freq_matches_huggingface():
    config = AutoConfig.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
    reference = LlamaRotaryEmbedding(config)  # pyright: ignore[reportArgumentType]
    rope = config.rope_parameters
    inv_freq = compute_rope_inv_freq(rope["rope_theta"], config.head_dim)
    inv_freq = apply_llama3_scaling(
        inv_freq,
        factor=config.rope_scaling["factor"],
        low_freq_factor=config.rope_scaling["low_freq_factor"],
        high_freq_factor=config.rope_scaling["high_freq_factor"],
        original_max_position_embeddings=config.rope_scaling["original_max_position_embeddings"]
    )
    assert_close(inv_freq, reference.inv_freq)