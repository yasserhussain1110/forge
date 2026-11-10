import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from forge.generate import generate
from forge.model import ForgeForCausalLM

model_id = "meta-llama/Llama-3.2-1B-Instruct"
config = AutoConfig.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

reference = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype=torch.float32,
    attn_implementation="eager"
)
model = ForgeForCausalLM(config)
model.load_state_dict(reference.state_dict())
del reference
model = model.to("cuda").eval()
eos_token_ids = [128001, 128008, 128009]
messages = [{"role": "user", "content": "What is the capital of France?"}]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
output_ids = generate(model, input_ids, eos_token_ids, max_new_tokens=40)
print(tokenizer.decode(output_ids[0], skip_special_tokens=True))

model = model.to("cpu")
torch.cuda.empty_cache()

hf = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype=torch.float32,
    attn_implementation="eager"
).to("cuda").eval()


with torch.no_grad():
    hf_ids = hf.generate(
        input_ids,
        max_new_tokens=40,
        do_sample=False,
        eos_token_id=eos_token_ids
    )
    print(tokenizer.decode(hf_ids[0], skip_special_tokens=True))