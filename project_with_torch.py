import util
from datasets import *

import torch
from torch.utils.data import Dataset

from torch import nn
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler

#Transforming data
wine_train_features = torch.from_numpy(WineData.X)
wine_train_labels = torch.from_numpy(WineData.Y)
wine_test_features = torch.from_numpy(WineData.Xte)
wine_test_labels = torch.from_numpy(WineData.Yte)

wine_train_features = wine_train_features.float()
wine_test_features = wine_test_features.float()
print(wine_train_features.dtype)
print(wine_train_labels.dtype)

class WineSet(Dataset):
    def __init__(self, X, Y, transform=None, target_transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        features = self.X[idx]
        label = self.Y[idx]

        if self.transform:
            features = self.transform(features)
        if self.target_transform:
            label = self.target_transform(label)
        return features, label

wine_train = WineSet(wine_train_features, wine_train_labels)
wine_test = WineSet(wine_test_features, wine_test_labels)
#features, label = wine_train[20]
#print(f"Features: \n{features}\n")
#print(f"Label: \n{label}\n")
#print(len(wine_train))
#print(features.shape)

train_dataloader = DataLoader(wine_train, batch_size=64)
test_dataloader = DataLoader(wine_test, batch_size=64)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.inp_layer = nn.Linear(in_features = 819, out_features = 20)
        self.stack = nn.Sequential(
                self.inp_layer,
#                nn.ReLU(),
#                nn.Linear(819, 20),
        )
        torch.nn.init.normal_(self.inp_layer.weight, mean=0.0, std=0.01)
    
    def forward(self, x):
        logits = self.stack(x)
        return logits

model = NeuralNetwork()

#for batch, (x, y) in enumerate(train_dataloader):
#    print(x)
#    print(y)
#    print(batch)
#    print("***************\n")

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (x, y) in enumerate(dataloader):
        pred = model(x)
        loss = loss_fn(pred, y)

        # Back propagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        #Print result of batch
        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(x)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluate with no_grad for no gradient computation
    with torch.no_grad():
        for x, y in dataloader:
            pred = model(x)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size

    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

learning_rate = 0.2
batch_size = 10
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
epochs = 50

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")

