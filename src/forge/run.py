from forge.model import load_llama
from transformers import AutoTokenizer
import torch
from forge.generate import generate
from huggingface_hub import hf_hub_download
import json

model_id = "meta-llama/Llama-3.2-1B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id, clean_up_tokenization_spaces=False)

llama = load_llama()

with open(hf_hub_download(model_id, "generation_config.json")) as f:
        config = json.load(f)

messages = [
    {"role": "user", "content": "Hello, What is the capital of France?"}
]

tokenized_chat = tokenizer.apply_chat_template(
    messages, tokenize=True,add_generation_prompt=True, return_tensors="pt")

with torch.no_grad():
    input_tokens = tokenized_chat["input_ids"][0].to("cuda")
    # input_tokens = tokenized_chat["input_ids"][0]
    response = generate(llama, input_tokens, 20, config["eos_token_id"])
    print(tokenizer.decode(response[len(input_tokens):], skip_special_tokens=True))