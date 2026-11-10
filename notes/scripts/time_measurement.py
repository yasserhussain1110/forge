import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from forge.model import ForgeForCausalLM
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from matplotlib import pyplot as plt

model_id = "meta-llama/Llama-3.2-1B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
config = AutoConfig.from_pretrained(model_id)
model = ForgeForCausalLM(config).to(dtype=torch.bfloat16)
weights = load_file(hf_hub_download(model_id, "model.safetensors"))
model.load_state_dict(weights, strict=False)
model = model.to("cuda").eval()

messages = [{"role": "user", "content": "How are you today?"}]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
generated = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")

step_times = []

with torch.no_grad():
    for _ in range(256):
        torch.cuda.synchronize()
        start = time.perf_counter()
        logits = model(generated)
        next_id = logits[0, -1].argmax()
        generated = torch.cat([generated, next_id.view(1,1)], dim=1)
        torch.cuda.synchronize()
        step_times.append(time.perf_counter() - start)

steps = list(range(1,257))

plt.plot(steps, step_times)
plt.savefig("notes/day-034-step-times.png")

print(256/sum(step_times))