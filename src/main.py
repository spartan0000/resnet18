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
from models import Block, ResNet


import argparse


#can specify model in command line arguments with defaul being resnet18
parser = argparse.ArgumentParser()
parser.add_argument('--model', type = str, default = 'resnet18', choices = ['resnet18', 'resnet34'])
args = parser.parse_args()


#environment variables
load_dotenv()
INPUT_DIR = 'D:/cifar10_preprocessed'
WANDB_API_KEY = os.environ.get('WANDB_API_KEY')

wandb.login(key = WANDB_API_KEY)


#path for saving state_dict at end of training loop
PATH = 'D:/resnet18_cifar10_2.pth'

#hyperparameters
EPOCHS = 70
batch_size = 128
lr = 0.1 * batch_size / 256
device = 'cuda' if torch.cuda.is_available() else 'cpu'
amp = True #automatic mixed precision for faster training on GPU with less memory usage
momentum = 0.9
weight_decay = 5e-4
label_smoothing = 0.1
subset_size = 25000
NUM_CLASSES = 10


#models
def cifar10_resnet18(num_classes = NUM_CLASSES):
    return ResNet(Block, [2,2,2,2], num_classes = NUM_CLASSES)

def cifar10_resnet34(num_classes = 10):
    return ResNet(Block, [3,4,6,3], num_classes = NUM_CLASSES)


#model dictionary for argument parsing - easier to add more models later
model_dict = {
    'resnet18': cifar10_resnet18,
    'resnet34': cifar10_resnet34,
}

#more data augmentations - these are used after the data loader since they are applied to batches
cut_mix = v2.CutMix(alpha = 1.0, num_classes= NUM_CLASSES)
mix_up = v2.MixUp(alpha = 1.0, num_classes = NUM_CLASSES)
cut_mix_or_mix_up = v2.RandomChoice([cut_mix, mix_up])

augmentations = ['random crop', 'random horizontal flip', 'normalization', 'cutmix or mixup']

#the model to be used as specified in command line arguments
net = model_dict[args.model](num_classes = NUM_CLASSES)


#optimizer, loss function, learning rate scheduler, scaler for mixed precision
optimizer = torch.optim.SGD(params = net.parameters(), lr = lr, momentum = momentum, weight_decay= weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2) 
loss_fn = nn.CrossEntropyLoss(label_smoothing = label_smoothing) #label smoothing to prevent overconfidence
scaler = torch.amp.GradScaler(device = device, enabled = amp)

#the exeriment data is the name of the experiment in wandb for easy tracking
experiment_date = datetime.now().strftime('%Y-%m-%d : %H-%M-%S') #use as experiment name in experiment tracking

#wandb experiment configuration
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
    'weight decay': weight_decay,
    'dropout': '0.5',
    'stochastic depth': 'yes, drop prob = 0.2',
    'data augmentations': augmentations,



}

#accuracy function - arguments are reversed order from usual scikit learn accuracy function

def accuracy(outputs, labels): #use inside training loop where 'outputs' are the the raw model outputs
    preds = torch.argmax(outputs, dim = 1)
    return torch.sum(preds == labels).item() /len(labels)


#training loop
def train(model, epochs, train_dataloader, test_dataloader, loss_fn, optimizer, device):
    wandb.init(
        project = f'{args.model} on CIFAR10',
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

            images, labels = cut_mix_or_mix_up(images, labels)

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
