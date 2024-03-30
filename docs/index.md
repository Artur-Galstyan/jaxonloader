# Jaxonloader

A dataloader that no one asked for, but here it is anyway.

### Yet another dataloader? Why...

Yes, it's true: the world may not need another dataloader, but I got tired of converting Torch tensors to JAX arrays and having this overhead everytime I want to train a model. So I decided to instead convert Numpy arrays to JAX and not having to do `tensor.numpy()` everytime I want to use the data.

### Ok, but what does this do?

It's a dataloader that is effectively the same as the PyTorch dataloader, except that it uses Numpy as the backend.

### So all you're doing is iterating over an array?

Yes. Do you really need more than that?
