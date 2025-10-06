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

idx_to_label = {
    0: 'airplane',
    1: 'automobile',
    2: 'bird',
    3: 'cat',
    4: 'deer',
    5: 'dog',
    6: 'frog',
    7: 'horse',
    8: 'ship',
    9: 'truck'
}

label_to_idx = {label:idx for idx, label in idx_to_label.items()}

input_dir = 'D:/cifar10_preprocessed'

model = ResNet18(Block, [2,2,2,2])

transform = v2.Compose([
    v2.Resize((32,32)),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale = True),
    v2.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

def model_inference(model, transform):
    model.load_state_dict(torch.load('D:/resnet18_cifar10.pth'))
    model.eval()

    with torch.inference_mode():

        label_df = pd.read_csv(os.path.join(input_dir, 'train_labels.csv'))

        random_nums = torch.randint(0,50000, (9,))
        label_idx = [n.item() for n in random_nums]
        image_idx = [f'{n:05d}' for n in random_nums]


        images = [Image.open(os.path.join(input_dir, f'train_img_{n}.jpg')) for n in image_idx]
        labels = [label_df['label'][n] for n in label_idx]

        predictions = []

        for img in images:
            transformed_img = transform(img).unsqueeze(0)
            output = model(transformed_img)
            probs = torch.softmax(output, dim = 1)
            preds = torch.argmax(probs, dim = 1)
            predictions.append(preds)

        classes = [idx_to_label.values()]

        fig, axs = plt.subplots(nrows = 3, ncols = 3, figsize = (8,8))
        for i, ax in enumerate(axs.flat):
            image = images[i]
            ax.imshow(image)
            ax.set_title({predictions[i].item()} == {labels[i]})
            ax.axis('off')
        plt.show()

if __name__ == '__main__':
    model_inference(model, transform)