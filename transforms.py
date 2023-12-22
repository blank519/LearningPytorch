import torch
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda
import matplotlib.pyplot as plt

ds = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
    target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
)

# Understanding the Lambda transform, assuming y = 5
print(f"zeros tensor: \n{torch.zeros(10, dtype=torch.float)}\n")

print(f"y tensor: \n{torch.tensor(5)}\n")

print(f"value?: \n{torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(5), value=3)}\n")

print(f"value?: \n{torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor([5, 3, 8, 1]), 3)}\n")

print(f"value?: \n{torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor([5, 3, 8, 1]), torch.tensor([1, 2, 3, 4], dtype=torch.float))}\n")

# Understanding the result of transforms

#img, label = ds[torch.randint(len(ds), size=(1,)).item()]
#print(f"Image: \n{img}\n")
#print(f"Label: \n{label}\n")

#figure = plt.figure(figsize=(8,8))
#plt.imshow(img.squeeze(), cmap="gray")
#plt.savefig("transform_sample.png")
