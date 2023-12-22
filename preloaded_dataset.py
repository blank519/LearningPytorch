import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

training_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor()
)

test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=False,
        transform=ToTensor()
)

labels_map = {
        0:"T-Shirt",
        1:"Trouser",
        2:"Pullover",
        3:"Dress",
        4:"Coat",
        5:"Sandal",
        6:"Shirt",
        7:"Sneaker",
        8:"Bag",
        9:"Ankle Boot",
}

figure = plt.figure(figsize=(8,8))
cols, rows = 4, 4

#for i in range(1, cols*rows + 1):
#    sample_idx = torch.randint(len(training_data), size=(1,)).item()
#    img, label = training_data[sample_idx]

#    figure.add_subplot(rows, cols, i)    
#    plt.title(labels_map[label])
#    plt.axis("off")
#    plt.imshow(img.squeeze(), cmap="gray")
#plt.savefig('train_sample.png')

#DataLoader
train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

# Display image and label.
train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")

for i in range(1, cols*rows+1):
    img = train_features[i].squeeze()
    label = train_labels[i]
    
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label.item()])
    #print(f"Label: {label}")
    plt.axis("off")
    plt.imshow(img, cmap="gray")
plt.savefig('loader_sample.png')
print(f"Label: {train_labels[0]}")
print(f"Normal Features:\n {train_features[0].shape}\n")
print(f"Squeezed Features:\n {train_features[0].squeeze().shape}\n")



