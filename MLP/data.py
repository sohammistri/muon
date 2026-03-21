import os
import zipfile
import urllib.request

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA_CACHE = os.path.join(os.path.dirname(__file__), "data_cache")


def get_covertype_loaders(batch_size=1024, seed=42):
    """Forest Covertype: 581K samples, 54 features, 7 classes."""
    data = fetch_covtype()
    X, y = data.data, data.target
    y = y - 1  # shift from 1-7 to 0-6

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    test_ds = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, 54, 7


def _download_year_prediction():
    """Download Year Prediction MSD dataset from UCI if not cached."""
    os.makedirs(DATA_CACHE, exist_ok=True)
    txt_path = os.path.join(DATA_CACHE, "YearPredictionMSD.txt")

    if os.path.exists(txt_path):
        return txt_path

    zip_path = os.path.join(DATA_CACHE, "yearpredictionmsd.zip")
    url = "https://archive.ics.uci.edu/static/public/203/yearpredictionmsd.zip"

    print(f"Downloading Year Prediction MSD from {url} ...")
    urllib.request.urlretrieve(url, zip_path)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(DATA_CACHE)

    os.remove(zip_path)
    return txt_path


def get_year_prediction_loaders(batch_size=1024, seed=42):
    """Year Prediction MSD: 515K samples, 90 features, regression."""
    txt_path = _download_year_prediction()
    data = np.loadtxt(txt_path, delimiter=",")

    # Canonical split: first 463715 train, last 51630 test
    X_train, y_train = data[:463715, 1:], data[:463715, 0]
    X_test, y_test = data[463715:, 1:], data[463715:, 0]

    feat_scaler = StandardScaler()
    X_train = feat_scaler.fit_transform(X_train)
    X_test = feat_scaler.transform(X_test)

    target_scaler = StandardScaler()
    y_train = target_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_test = target_scaler.transform(y_test.reshape(-1, 1)).ravel()

    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    test_ds = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, 90, 1


def get_mnist_loaders(batch_size=1024, seed=42):
    """MNIST: 60K train / 10K test, 784 features (flattened 28x28), 10 classes."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Lambda(lambda x: x.view(-1)),
    ])

    train_ds = datasets.MNIST(DATA_CACHE, train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(DATA_CACHE, train=False, download=True, transform=transform)

    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=g)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, 784, 10
