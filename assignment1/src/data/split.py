#split.py
'''
This module contains a function to create a train-validation split from a dataset.
We will split the training dataset into a training and validation set (10% of the training data), while keeping the test dataset separate.
'''

import numpy as np
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split

def create_train_val_split(labels, validation_size, seed):
    indices = np.arange(len(labels))

    train_indices, val_indices = train_test_split(
        indices,
        test_size = validation_size,
        random_state = seed,
        stratify = labels,
    )

    return train_indices, val_indices

def create_train_val_datasets(train_dataset, val_dataset, train_indices, val_indices):
    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(val_dataset, val_indices)

    return train_subset, val_subset

