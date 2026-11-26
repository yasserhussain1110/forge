import torch
from forge.model import Llama

def generate(model: Llama, input_tokens, eos_tokens):
    max_new_tokens = model.max_sequence_length - len(input_tokens)
    start = 0
    model.reset_kv_cache()
    for _ in range(max_new_tokens):
        response = torch.argmax(model(input_tokens[start:,])[-1])
        start = input_tokens.size()[0]
        if response in eos_tokens:
            break
        input_tokens = torch.cat((input_tokens, response[None,...]), dim=0)
    return input_tokens