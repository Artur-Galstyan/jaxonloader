# The Future plans for Jaxonloader

Jaxonloader is still a small baby, but it has big dreams.

At its core, I think, we will keep the `DataFrame` -> `JaxonDataset` -> `JaxonDataLoader` pipeline.

In terms of additional features, I'm thinking about **not** adding any preprocessing steps, i.e. BYOPP (bring your own preprocessing).

On the other hand, it'd be nice if we could add all kinds of "standard" ML datasets, decoupling ourselves from the PyTorch dataloader.
