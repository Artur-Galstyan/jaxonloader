from datasets import load_dataset


dataset = load_dataset("rotten_tomatoes")
print(type(dataset))
print(dataset["train"])
print(dataset["train"][0])
