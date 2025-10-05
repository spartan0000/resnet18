#small script for inference using saved model

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

from models import Block, ResNet18

input_dir = 'D:/cifar10_preprocessed'

model = ResNet18(Block, [2,2,2,2])

transform = transforms.Compose([
    v2.Resize((32,32)),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale = True),
    v2.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

def model_inference(model, transform):
    model.load_state_dict(torch.load('resnet18_cifar10.pth'))
    model.eval()

    with torch.inference_mode():
        images = [Image.open(os.path.join(input_dir, f'img_{i}.jpg')) for i in range(4)]
        tensors = torch.stack(transform(img) for img in images)

        output = model(tensors)
        probs = torch.softmax(output, dim = 1)
        preds = torch.argmax(probs, dim = 1)

        

