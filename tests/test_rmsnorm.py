import torch
from torch.testing import assert_close
from transformers.models.llama.modeling_llama import LlamaRMSNorm

from forge.rmsnorm import RMSNorm

def test_rmsnorm_matches_huggingface():
    torch.manual_seed(0)
    hidden_size = 16
    eps = 1e-5
    reference = LlamaRMSNorm(hidden_size, eps=eps)
    mine = RMSNorm(hidden_size, eps=eps)
    mine.load_state_dict(reference.state_dict())

    x = torch.randn(2, 4, hidden_size)
    with torch.no_grad():
        assert_close(mine(x), reference(x))