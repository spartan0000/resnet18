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

def cifar10_resnet18(num_classes = 10):
    return ResNet18(Block, [2,2,2,2], num_classes = 10)

net = cifar10_resnet18()

def train():
    pass



def main():
    train_dataloader, test_dataloader, train_subsetloader, test_subsetloader = build_dataloaders(INPUT_DIR, subset_size = 2000, batch_size = 32)


if __name__ == "__main__":
    main()
