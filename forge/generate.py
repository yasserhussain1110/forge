import torch

def generate(model, input_ids, eos_token_ids, max_new_tokens=40):
    generated = input_ids
    eos_token_ids = set(eos_token_ids)
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(generated)
            next_id = logits[0, -1].argmax()
            generated = torch.cat(
                [generated, next_id.view(1, 1)],
                dim=1
            )
            if next_id.item() in eos_token_ids:
                break
    return generated
