#dataset.py

from pathlib import Path

from torchvision.datasets import FashionMNIST

def get_fashion_mnist_datasets(
    data_dir: str | Path,
    train_transforms = None,
    val_transforms = None,
    test_transforms = None,
):
    data_dir = Path(data_dir)

    train_dataset = FashionMNIST(
        root=data_dir,
        train=True,
        download=True,
        transform = train_transforms,
    )

    val_dataset = FashionMNIST(
        root=data_dir,
        train=True,
        download=True,
        transform = val_transforms,
    )

    test_dataset = FashionMNIST(
        root=data_dir,
        train=False,
        download=True,
        transform = test_transforms,
    )

    return train_dataset, val_dataset, test_dataset
