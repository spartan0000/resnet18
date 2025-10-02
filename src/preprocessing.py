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
    #uses huggingface datsets library to download cifar10 dataset
    """
    output_dir: directory where the images and labels will be saved
    dataset_name: name of the dataset to be downloaded from huggingface datasets library
    """
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
    labels = [img_labels['label'][n] for n in random_idx]

    fig, axes = plt.subplots(3,3, figsize = (8,8))

    for i, ax in enumerate(axes.flat):
        img = imgs[i]
        label = labels[i]

        ax.imshow(img)
        ax.set_title(idx_to_label[label])
        ax.axis('off')
    plt.tight_layout()
    plt.show()


def get_filenames_labels(input_dir, dataset):
    """
    input_dir: directory where the images are stored
    dataset: 'train' or 'test'
    """
    filenames = [f for f in os.scandir(input_dir) if f.name.endswith('.jpg') and f.name.startswith(f'{dataset}')]
    labels_df = pd.read_csv(os.path.join(input_dir, f'{dataset}_labels.csv'))
    labels = labels_df['label'].tolist()

    return filenames, labels


class Cifar10Dataset(Dataset):
    def __init__(self, input_dir, filenames, labels, transform = None):
        self.input_dir = input_dir
        self.filenames = filenames
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        img_path = os.path.join(self.input_dir, filename)
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

def build_dataloaders(input_dir, subset_size, batch_size = 32):
    """
    input_dir: directory where the images and labels are stored
    subset_size: size of the subset to be used for training and testing if necessary
    batch_size: batch size for dataloaders
    """

    #if the subset is not necessary, will take this out later
    
    train_filenames, train_labels = get_filenames_labels(input_dir, 'train')
    test_filenames, test_labels = get_filenames_labels(input_dir, 'test')

    transform = v2.Compose([
        v2.RandomCrop((32, 32), padding = 4),
        v2.RandomHorizontalFlip(),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale = True),
        v2.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), #mean and std for RGB channels
    ])

    train_dataset = Cifar10Dataset(input_dir, train_filenames, train_labels, transform = transform)
    test_dataset = Cifar10Dataset(input_dir, test_filenames, test_labels, transform = transform)

    train_subset = Subset(train_dataset, range(subset_size))
    test_subset = Subset(test_dataset, range(subset_size))

    train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    test_dataloader = DataLoader(test_dataset, batch_size = batch_size)

    train_subsetloader = DataLoader(train_subset, batch_size = batch_size, shuffle = True)
    test_subsetloader = DataLoader(test_subset, batch_size = batch_size)

    return train_dataloader, test_dataloader, train_subsetloader, test_subsetloader