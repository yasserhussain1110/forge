What bugs did I hit, and how I found them?

1. Loading multiple models in the same gpu when my gpu mem is 8GB.
Fix - Unload one model before loading the next

What is my baseline speed?
10 tokens/second

What was confusing this week?

1. What is model? Is it a process or a file?
    It is both. A model like llama is a model file with weights. I can download it and load it in my GPU, but it won't do anything yet.
    It's the transformer library which will load the model and then allow me to interact with it. So in this regard the after the library loads the file, that becomes the model because that is what I interact with.
2. What's the role of torch.tensors in all this?
    The results returned by the model process are vectors represented as torch tensor objects.