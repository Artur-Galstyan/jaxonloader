## Installation

Install this package using pip like so:

`pip install jaxonloader`

## Usage

The _intended_ way to use `Jaxonloader` is the following:

1. You start with a `DataFrame`
2. You preprocess your data in that `DataFrame`
3. You convert that `DataFrame` into a `JaxonDataset`
4. You pass that `JaxonDataset` into a `JaxonDataLoader`
5. ???
6. Profit

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
jaxon_dataloader, state = make(
    jaxon_dataset, batch_size=64, shuffle=True, key=subkey, jit=True
)

# Step 5
while it := jaxon_dataloader(state):
    x, state, done = it
    if done:
        break

# Step 6
print('Profit!')

```
