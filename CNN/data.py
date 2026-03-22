import os

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

DATA_CACHE = os.path.join(os.path.dirname(__file__), "data_cache")

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def get_cifar10_loaders(batch_size=128, seed=42):
    """CIFAR-10: 50K train / 10K test, 3×32×32 images, 10 classes."""
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    train_ds = datasets.CIFAR10(DATA_CACHE, train=True, download=True,
                                transform=train_transform)
    test_ds = datasets.CIFAR10(DATA_CACHE, train=False, download=True,
                               transform=test_transform)

    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              generator=g, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=2, pin_memory=True)

    return train_loader, test_loader, 3, 10
