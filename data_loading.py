import os

import numpy as np
import torch

from numpy import genfromtxt
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset
from torchvision.io import read_image
from torchvision.transforms import Compose, ConvertImageDtype


class ImageDataset(VisionDataset):
    """
    Dataset class for images
    """

    def __init__(self, root, transforms=None, target_transform=None):
        super(ImageDataset, self).__init__(root, transform=transforms, target_transform=target_transform)
        self.root = root
        self.transform = transforms
        self.target_transform = target_transform

        self.images_names = [x.replace(".jpg", "") for x in os.listdir(root)]
        self.targets = {
            name: age for name, age in genfromtxt(
                os.path.join("data", os.path.dirname(root).split("/")[-1] + ".csv"),
                delimiter=',',
                dtype=str,
                skip_header=1)
        }

    def __getitem__(self, index):
        """
        Returns the image and its target
        """
        name = self.images_names[index]
        target = self.targets[name]
        img = read_image(os.path.join(self.root, name) + ".jpg")
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.images_names)


def load_datasets(batch_size, num_workers=0):
    """
    Loads the datasets
    """
    train_set = ImageDataset(
        root="data/train/",
        transforms=get_image_transforms(),
        target_transform=target_transform
    )
    val_set = ImageDataset(
        root="data/val/",
        transforms=get_image_transforms(),
        target_transform=target_transform
    )
    test_set = ImageDataset(
        root="data/test/",
        transforms=get_image_transforms(),
        target_transform=target_transform
    )
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, val_loader, test_loader


def get_image_transforms():

    return Compose([ConvertImageDtype(torch.float)])


def target_transform(target):
    """
    Transform target to a tensor
    """
    target = [int(x) for x in target.split("-")]
    target = np.mean(target)
    return torch.tensor(target).float()
