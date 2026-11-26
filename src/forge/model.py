from torch import nn
from forge.rmsnorm import RMSNorm
from forge.decoder import Decoder
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
import json

model_id = "meta-llama/Llama-3.2-1B-Instruct"

class Llama(nn.Module):
    def __init__(self, hidden_size, intermediate_size, num_attention_heads, num_key_value_heads, head_dim,
                 rope_theta, rms_norm_eps, vocab_size, num_hidden_layers, max_sequence_length):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([
            Decoder(hidden_size, intermediate_size, num_attention_heads, num_key_value_heads, head_dim, rope_theta,
                    rms_norm_eps, max_sequence_length) for _ in range(num_hidden_layers)])
        self.norm = RMSNorm(hidden_size, rms_norm_eps)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.lm_head.weight = self.embed_tokens.weight
        self.max_sequence_length = max_sequence_length

    def forward(self, token_ids):
        x = self.embed_tokens(token_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.lm_head(x)

    def reset_kv_cache(self):
        for layer in self.layers:
            layer.reset_kv_cache()

def load_llama(max_sequence_length):
    weights = {
        k.removeprefix("model."):v.float() for k,v in load_file(hf_hub_download(model_id, "model.safetensors")).items()
    }
    with open(hf_hub_download(model_id, "config.json")) as f:
        config = json.load(f)
    llama = Llama(
        config["hidden_size"], config["intermediate_size"], config["num_attention_heads"], config["num_key_value_heads"],
        config["head_dim"], config["rope_theta"], config["rms_norm_eps"], config["vocab_size"], config["num_hidden_layers"],
        max_sequence_length
    )
    llama.load_state_dict(weights, strict=False)
    return llama.to("cuda").eval()
    # return llama.eval()