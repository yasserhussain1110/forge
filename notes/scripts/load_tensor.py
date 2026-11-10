from safetensors import safe_open

path = "/home/yasser/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/921\
3176726f574b556790deb65791e0c5aa438b6/model.safetensors"

f = safe_open(path, framework="pt")

for _, tensor in f.get_tensors().items():
    print(f"{tensor.name}, {tensor.shape}, {tensor.dtype}")
