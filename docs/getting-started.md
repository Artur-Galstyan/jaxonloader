## Installation

Install this package using pip like so:

`pip install jaxonloader`

## Usage

It's quite similar to the PyTorch dataloader.

1. Create a dataset which inherits from `jaxonloader.JaxonDataset`.
2. Implement all the abstract methods.
3. Create a dataloader using `jaxonloader.JaxonDataLoader`.
4. Iterate over the dataloader.
5. ???
6. Profit!

Here's an example:

```python

import pandas as pd
from jaxonloader import make, from_dataframe

# Step 1
df = pd.read_csv('data.csv')

# Step 2
df['column'] = df['column'].apply(lambda x: x + 1)

# Step 3
jaxon_dataset = from_dataframe(df)

# Step 4
jaxon_dataloader = JaxonDataLoader(
    jaxon_dataset, batch_size=64, shuffle=True, key=subkey, jit=True
)

# Step 5
for x in jaxon_dataloader:
    pass

# Step 6
print('Profit!')

```
