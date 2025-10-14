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
from timeit import default_timer as timer   

import wandb #weights and biases for experiment tracking
from dotenv import load_dotenv
import os

from datetime import datetime


#imports

from preprocessing import get_datasets, preprocess_data, plot_samples, get_filenames_labels, build_dataloaders, Cifar10Dataset
from models import Block, ResNet18

load_dotenv()
INPUT_DIR = 'D:/cifar10_preprocessed'
WANDB_API_KEY = os.environ.get('WANDB_API_KEY')

wandb.login(key = WANDB_API_KEY)

PATH = 'D:/resnet18_cifar10_2.pth'

#hyperparameters
EPOCHS = 70
lr = 1e-1
device = 'cuda' if torch.cuda.is_available() else 'cpu'
amp = True #automatic mixed precision for faster training on GPU with less memory usage
batch_size = 128
subset_size = 25000
momentum = 0.9
weight_decay = 1e-3
label_smoothing = 0.1

experiment_date = datetime.now().strftime('%Y-%m-%d : %H-%M-%S') #use as experiment name in experiment tracking

config = {
    'learning rate': lr,
    'learning rate decay': 'cosine annealing warm restarts',
    'epochs': EPOCHS,
    'batch size' : batch_size,
    'optimizer': 'SGD',
    'momentum': momentum,    
    'loss function': 'CrossEntropyLoss with label smoothing',
    'model': 'ResNet18',
    'dataset': 'CIFAR10',
    'device': device,
    'amp': amp,
    'subset': 'no',
    'subset size': 'N/A',
    'weight decay': weight_decay,
    'dropout': '0.5',
    'stochastic depth': 'no',


}


def cifar10_resnet18(num_classes = 10):
    return ResNet18(Block, [2,2,2,2], num_classes = 10)

net = cifar10_resnet18()
optimizer = torch.optim.SGD(params = net.parameters(), lr = lr, momentum = momentum, weight_decay= weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2) 
loss_fn = nn.CrossEntropyLoss(label_smoothing = label_smoothing) #label smoothing to prevent overconfidence
scaler = torch.amp.GradScaler(device = device, enabled = amp)





def accuracy(outputs, labels): #use inside training loop where 'outputs' are the the raw model outputs
    preds = torch.argmax(outputs, dim = 1)
    return torch.sum(preds == labels).item() /len(labels)

def train(model, epochs, train_dataloader, test_dataloader, loss_fn, optimizer, device):
    wandb.init(
        project = 'ResNet18 on CIFAR10',
        name = f'{experiment_date}',
        config = config,
    )
    model.to(device)
    start = timer()
    best_test_acc = 0.0
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for i, (images,labels) in enumerate(train_dataloader):

            images, labels = images.to(device), labels.to(device)
            with torch.autocast(device_type = device, enabled = amp):
                outputs = model(images)
                loss = loss_fn(outputs, labels)

            optimizer.zero_grad()
            #torch.autograd.set_detect_anomaly(True)
            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()
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
            if test_acc > best_test_acc:
                best_test_acc = test_acc
        scheduler.step()
            
        print(f'Epoch {epoch+1}/{epochs}: Train loss: {train_loss:.4f}| Test loss: {test_loss:.4f}| Test accuracy: {test_acc:.4f}')

        wandb.log({
            'epoch': epoch,
            'train_loss': train_loss,
            'test loss': test_loss,
            'test accuracy': test_acc,
            'best test accuracy': best_test_acc,


        })
   
    end = timer()

    print(f'Total training time on {device}: {end - start:.2f} seconds')
    torch.save(model.state_dict(), PATH)

def main():
    train_dataloader, test_dataloader, train_subsetloader, test_subsetloader = build_dataloaders(INPUT_DIR, subset_size = subset_size, batch_size = batch_size)
    
    train(net, EPOCHS, train_dataloader, test_dataloader, loss_fn, optimizer, device)

if __name__ == "__main__":
    main()
