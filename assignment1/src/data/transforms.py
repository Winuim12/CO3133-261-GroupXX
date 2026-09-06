#transforms.py
'''
Each sample in FashionMNIST is a 28x28 grayscale image
with pixel values in [0, 255] and a label.

This module transforms each sample into a PyTorch tensor
with shape [C, H, W]. For grayscale images, the shape is
[1, 28, 28]. ToTensor() also scales uint8 pixel values
from [0, 255] to [0.0, 1.0].
'''

from torchvision import transforms

def get_train_transforms(mean=None, std=None):
    transforms_list = [
        transforms.ToTensor(),
    ]

    if mean is not None and std is not None:
        transforms_list.append(
            transforms.Normalize(mean=(mean,), std=(std,))
        )

    return transforms.Compose(transforms_list)

def get_val_transforms(mean=None, std=None):
    transforms_list = [
        transforms.ToTensor(),
    ]

    if mean is not None and std is not None:
        transforms_list.append(
            transforms.Normalize(mean=(mean,), std=(std,))
        )

    return transforms.Compose(transforms_list)

def get_test_transforms(mean=None, std=None):
    transforms_list = [
        transforms.ToTensor(),
    ]

    if mean is not None and std is not None:
        transforms_list.append(
            transforms.Normalize(mean=(mean,), std=(std,))
        )

    return transforms.Compose(transforms_list)