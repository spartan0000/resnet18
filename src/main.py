import torch

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from torch import nn

import torchvision
from torchvision import datasets
from torchvision.transforms import v2

from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image


#imports

from preprocessing import get_datasets, preprocess_data, plot_samples, get_filenames_labels, build_dataloaders, Cifar10Dataset
from models import Block, ResNet18


INPUT_DIR = 'D:/cifar10_preprocessed'

#hyperparameters
EPOCHS = 5
lr = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
amp = True 


def cifar10_resnet18(num_classes = 10):
    return ResNet18(Block, [2,2,2,2], num_classes = 10)

net = cifar10_resnet18()
optimizer = torch.optim.Adam(params = net.parameters(), lr = lr)
loss_fn = nn.CrossEntropyLoss()
scaler = torch.amp.GradScaler(device = device, enabled = amp)


def accuracy(outputs, labels):
    preds = torch.argmax(outputs, dim = 1)
    return torch.sum(preds == labels).item() /len(labels)

def train(model, epochs, train_dataloader, test_dataloader, loss_fn, optimizer, device):
    pass

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for i, (images,labels) in enumerate(train_dataloader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            optimizer.zero_grad()

            loss.backward()

            optimizer.step()
            train_loss += loss.item() * images.size(0)
        train_loss /= len(train_dataloader.dataset)

        model.eval()
        with torch.inference_mode():
            test_loss = 0.0
            test_acc = 0.0
            for i, (images, labels) in enumerate(test_dataloader):
                images, labels = images.to(device), labels.to(device)
                test_outputs = model(images)
                t_loss = loss_fn(test_outputs, labels)
                test_loss += t_loss.item() * images.size(0)
                test_acc += accuracy(test_outputs, labels) * images.size(0)
            test_loss /= len(test_dataloader.dataset)
            test_acc /= len(test_dataloader.dataset)
        print(f'Epoch {epoch+1}/{epochs}: Train loss: {train_loss:.4f}| Test loss: {test_loss:.4f}| Test accuracy: {test_acc:.4f}')



def main():
    train_dataloader, test_dataloader, train_subsetloader, test_subsetloader = build_dataloaders(INPUT_DIR, subset_size = 2000, batch_size = 32)


if __name__ == "__main__":
    main()
