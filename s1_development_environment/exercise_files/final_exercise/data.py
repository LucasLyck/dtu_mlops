import torch
import numpy as np
from torchvision import datasets, transforms
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

def mnist():

    dir_root = Path(__file__).parent.parent.parent.parent
    train_path_1 = Path(dir_root, "./data/corruptmnist/train_0.npz")
    train_path_2 = Path(dir_root, "./data/corruptmnist/train_1.npz")
    train_path_3 = Path(dir_root, "./data/corruptmnist/train_2.npz")
    train_path_4 = Path(dir_root, "./data/corruptmnist/train_3.npz")
    train_path_5 = Path(dir_root, "./data/corruptmnist/train_4.npz")

    test_path = Path(dir_root, "./data/corruptmnist/test.npz")

    with np.load(train_path_1) as data:
        train_images_1 = data['images']
        train_labels_1 = data['labels']

    with np.load(train_path_2) as data:
        train_images_2 = data['images']
        train_labels_2 = data['labels']
    with np.load(train_path_3) as data:
        train_images_3 = data['images']
        train_labels_3 = data['labels']

    with np.load(train_path_3) as data:
        train_images_3 = data['images']
        train_labels_3 = data['labels']

    with np.load(train_path_4) as data:
        train_images_4 = data['images']
        train_labels_4 = data['labels']
    
    with np.load(train_path_5) as data:
        train_images_5 = data['images']
        train_labels_5 = data['labels']


    with np.load(test_path) as data:
        test_images = data['images']
        test_labels = data['labels']
    
    train_images = np.concatenate((train_images_1, train_images_2, train_images_3, train_images_4, train_images_5))
    train_labels = np.concatenate((train_labels_1, train_labels_2, train_labels_3, train_labels_4, train_labels_5))
    class ImageDataset(Dataset):
        def __init__(self, images, labels):
            self.images = images
            self.labels = labels

        def __len__(self):
            return len(self.images)

        def __getitem__(self, idx):
            return self.images[idx], self.labels[idx]

    train_dataset = ImageDataset(train_images, train_labels)
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    test_dataset = ImageDataset(test_images, test_labels)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True)

    return train_dataloader, test_dataloader


