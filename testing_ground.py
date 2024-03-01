import time

from jaxonloader import get_mnist
from jaxonloader.dataloader import DataLoader as JaxonDataLoader
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)
dataset1 = datasets.MNIST("../data", train=True, download=True, transform=transform)
train_loader = DataLoader(dataset1, batch_size=64, shuffle=True)
print("Starting to iterate through training data...")
start_time = time.time()

for data in train_loader:
    pass

print(f"Time to iterate through training data: {time.time() - start_time:.2f} seconds")

train, test = get_mnist()
jaxon_train_loader = JaxonDataLoader(train, batch_size=64)

print("Starting to iterate through training data...")
start_time = time.time()

for data in jaxon_train_loader:
    pass

print(f"Time to iterate through training data: {time.time() - start_time:.1f} seconds")
