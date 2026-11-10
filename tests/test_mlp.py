import torch
from torch.testing import assert_close
from transformers.models.llama.modeling_llama import LlamaMLP

from forge.mlp import MLP

class TinyConfig:
    hidden_size = 16
    intermediate_size = 32
    mlp_bias = False
    hidden_act = "silu"

def test_mlp_matches_huggingface():
    torch.manual_seed(0)
    config = TinyConfig()
    reference = LlamaMLP(config)
    mine = MLP(config.hidden_size, config.intermediate_size)
    mine.load_state_dict(reference.state_dict())
    x = torch.randn(2, 4, config.hidden_size)
    with torch.no_grad():
        assert_close(mine(x), reference(x))