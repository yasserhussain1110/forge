import torch

def generate(model, input_tokens, max_new_tokens, eos_tokens):
    for _ in range(max_new_tokens):
        response = torch.argmax(model(input_tokens)[-1])
        if response in eos_tokens:
            break
        input_tokens = torch.cat((input_tokens, response[None,...]), dim=0)
    return input_tokens