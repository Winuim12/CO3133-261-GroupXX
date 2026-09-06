#statistics.py

import torch

def calculate_mean_std(dataset):
    '''
    Calculates only correct if dataset reutrns images as tensors with pixel values in [0.0, 1.0].
    '''
    pixel_sum = 0.0
    pixel_squared_sum = 0.0 
    num_pixels = 0

    for image, _ in dataset:
        pixel_sum += image.sum().item()
        pixel_squared_sum += (image ** 2).sum().item()
        num_pixels += image.numel()

    mean = pixel_sum / num_pixels
    variance = (pixel_squared_sum / num_pixels) - (mean ** 2)
    std = variance ** 0.5

    return mean, std
    