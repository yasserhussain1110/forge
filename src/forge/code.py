import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "meta-llama/Llama-3.2-1B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id, clean_up_tokenization_spaces=False)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cuda", dtype=torch.float32)

messages = [
    {"role": "user", "content": "Hello, how are you?"}
]

tokenized_chat = tokenizer.apply_chat_template(
    messages, tokenize=True,add_generation_prompt=True, return_tensors="pt").to("cuda")

outputs = model.generate(**tokenized_chat, max_new_tokens=128, do_sample=False)
response = outputs[:,tokenized_chat['input_ids'].size()[1]:]

print(tokenizer.decode(response[0], skip_special_tokens=True))