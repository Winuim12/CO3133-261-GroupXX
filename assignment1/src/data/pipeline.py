#pipeline.py
'''

'''
from src.data.dataset import get_fashion_mnist_datasets
from src.data.split import create_train_val_split, create_train_val_datasets
from src.data.transforms import get_train_transforms, get_val_transforms, get_test_transforms
from src.data.statistics import calculate_mean_std

def create_normalized_datasets(data_dir, validation_size, seed):
    train_dataset, val_dataset, _ = get_fashion_mnist_datasets(
        data_dir=data_dir,
        train_transforms=get_train_transforms(),
        val_transforms=get_val_transforms(),
        test_transforms=get_test_transforms(),
    )

    train_indices, val_indices = create_train_val_split(
        labels=train_dataset.targets,
        validation_size=validation_size,
        seed=seed,
    )

    train_subset, val_subset = create_train_val_datasets(
        train_dataset,
        val_dataset,
        train_indices,
        val_indices,
    )

    mean, std = calculate_mean_std(train_subset)

    normalized_train_dataset, normalized_val_dataset, test_dataset =(
        get_fashion_mnist_datasets(
            data_dir=data_dir,
            train_transforms=get_train_transforms(mean, std),
            val_transforms=get_val_transforms(mean, std),
            test_transforms=get_test_transforms(mean, std),
        )
    )

    normalized_train_subset, normalized_val_subset = (
        create_train_val_datasets(
            normalized_train_dataset,
            normalized_val_dataset,
            train_indices,
            val_indices,
        )
    )

    return normalized_train_subset, normalized_val_subset, test_dataset, mean, std