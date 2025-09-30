import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch import nn

import torchvision
from torchvision import datasets
from torchvision.transforms import v2

from torch.utils.data import Dataset, DataLoader
from PIL import Image

from datasets import load_dataset

import os

#cifar 10 already split into training and testing sets

#labels
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





def get_datasets(output_dir: str, dataset_name: str):
    dataset = load_dataset(dataset_name, cache_dir = output_dir)
    train_dataset = dataset['train']
    test_dataset = dataset['test']

    return train_dataset, test_dataset


def preprocess_data(train_dataset, test_dataset, output_dir): #take dataset, preprocess images, save to output directory, save labels in csv file for future use
    #transform not necessary here since all the images are the same size but included for datasets with varying image sizes
    transform = v2.Compose([
        v2.Resize((32,32)),
    ])

    train_records = []
    test_records = []

    for i, example in enumerate(train_dataset): #save each image to an output directory with a filename that includes its index in the dataset, save labels in a csv file
        img = example['image']
        label = example['label']

        img = transform(img)

        filename = os.path.join(output_dir, f'train_img_{i:05d}.jpg')
        img.save(filename)
        train_records.append({'filename': filename, 'label': label})
    pd.DataFrame(train_records).to_csv(os.path.join(output_dir, 'train_labels.cxv'), index = False)

    for i, example in enumerate(test_dataset): #same as above but for test set
        img = example['image']
        label = example['label']

        img = transform(img)

        filename = os.path.join(output_dir, f'test_img_{i:05d}.jpg')
        img.save(filename)
        test_records.append({'filename': filename, 'label': label})
    pd.DataFrame(test_records).to_csv(os.path.join(output_dir, 'test_labels.csv'), index = False)


                             
def plot_samples(input_dir: str, image_set: str, n_samples: int = 9):
    """
    Plot random samples from the dataset.  
    input_dir: directory where the images are stored
    image_set: 'train' or 'test'
    n_samples: number of random samples - set to 9 by default.  May have to change size of figure if this is changed
    """

    random_nums = torch.randint(0,50000, (n_samples,))
    random_idx = [n.item() for n in random_nums] #index to get labels since the labels exist in csv file where the indices are not padded to 5 digits
    random_num_padded = [f'{n:05d}' for n in random_idx] #need padded indices to get images using their filenames

    imgs = [Image.open(os.path.join(input_dir, f'{image_set}_img_{n}.jpg')) for n in random_num_padded]
    img_labels = pd.read_csv(os.path.join(input_dir, f'{image_set}_labels.csv'))
    labels = [img_labels['label'[n] for n in random_idx]]

    fig, axes = plt.subplots(3,3, figsize = (8,8))

    for i, ax in enumerate(axes.flat):
        img = imgs[i]
        label = labels[i]

        ax.imshow(img)
        ax.set_title(idx_to_label[label])
        ax.axis('off')
    plt.tight_layout()
    plt.show()


