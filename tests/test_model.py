import torch
from torch.testing import assert_close
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from forge.model import ForgeForCausalLM

def test_logits_match_huggingface():
    model_id = "meta-llama/Llama-3.2-1B-Instruct"
    config = AutoConfig.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    reference = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32, attn_implementation="eager")
    mine = ForgeForCausalLM(config)
    missing, unexpected = mine.load_state_dict(reference.state_dict(), strict=False)
    print("missing", missing)
    print("unexpected", unexpected)
    # device = "cuda"
    # reference = reference.to(device).eval()
    # mine = mine.to(device).eval()
    reference = reference.eval()
    mine = mine.eval()
    messages = [{"role": "user", "content": "Say hello in one sentence."}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    with torch.no_grad():
        ref_logits = reference(input_ids=input_ids).logits
        mine_logits = mine(input_ids)
    max_diff = (ref_logits - mine_logits).abs().max().item()
    print("max logit difference", max_diff)
    assert max_diff < 1e-3
    ref_top = ref_logits[0, -1].argmax()
    mine_top = mine_logits[0, -1].argmax()
    assert ref_top.item() == mine_top.item()