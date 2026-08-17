"""
Dataset loaders for TTN experiments.
Supports MNIST, Fashion-MNIST, and CIFAR-10 with proper preprocessing.
"""

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from typing import Tuple, Optional, Dict, Any


# ---------------------------------------------------------------------------
# Preprocessing pipelines
# ---------------------------------------------------------------------------

class Flatten:
    """Flatten transform (picklable, unlike lambda)."""
    def __call__(self, x):
        return x.view(-1)


def _get_transforms(dataset_name: str, resize: Optional[int] = None) -> transforms.Compose:
    """Build preprocessing transforms for a given dataset."""
    transform_list = []

    if resize is not None:
        transform_list.append(transforms.Resize(resize))

    transform_list.append(transforms.ToTensor())

    # Normalize to [0, 1] — ToTensor already does this for PIL images
    # For TTN feature maps, we want values in [0, 1]
    # No further normalization needed (feature maps handle the embedding)

    # Flatten for tensor network input (use class instead of lambda for pickling)
    transform_list.append(Flatten())

    return transforms.Compose(transform_list)


# ---------------------------------------------------------------------------
# Dataset factory
# ---------------------------------------------------------------------------

_DATASET_REGISTRY = {
    "mnist": datasets.MNIST,
    "fashion_mnist": datasets.FashionMNIST,
    "cifar10": datasets.CIFAR10,
}

_DATASET_INFO = {
    "mnist": {"num_classes": 10, "input_dim": 784, "channels": 1},
    "fashion_mnist": {"num_classes": 10, "input_dim": 784, "channels": 1},
    "cifar10": {"num_classes": 10, "input_dim": 3072, "channels": 3},
}


def get_dataset(
    name: str,
    root: str = "./data",
    resize: Optional[int] = None,
) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset, Dict[str, Any]]:
    """
    Load train and test datasets.

    Args:
        name: Dataset name ('mnist', 'fashion_mnist', 'cifar10')
        root: Root directory for data download
        resize: Optional resize dimension

    Returns:
        (train_dataset, test_dataset, info_dict)
    """
    if name not in _DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset: {name}. Choose from {list(_DATASET_REGISTRY.keys())}")

    dataset_cls = _DATASET_REGISTRY[name]
    transform = _get_transforms(name, resize)

    train_dataset = dataset_cls(root=root, train=True, download=True, transform=transform)
    test_dataset = dataset_cls(root=root, train=False, download=True, transform=transform)

    info = _DATASET_INFO[name].copy()
    if resize is not None:
        channels = info["channels"]
        info["input_dim"] = channels * resize * resize

    return train_dataset, test_dataset, info


def get_dataloaders(
    name: str,
    root: str = "./data",
    batch_size: int = 256,
    num_workers: int = 4,
    resize: Optional[int] = None,
    max_samples: Optional[int] = None,
    val_split: float = 0.1,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, Any]]:
    """
    Get train, validation, and test dataloaders.

    Args:
        name: Dataset name
        root: Root directory
        batch_size: Batch size
        num_workers: Number of data loading workers
        resize: Optional resize
        max_samples: Limit training samples (for debugging)
        val_split: Fraction of training data for validation
        seed: Random seed for reproducible splits

    Returns:
        (train_loader, val_loader, test_loader, info_dict)
    """
    train_dataset, test_dataset, info = get_dataset(name, root, resize)

    # Create train/val split
    num_train = len(train_dataset)
    if max_samples is not None:
        num_train = min(num_train, max_samples)

    num_val = int(num_train * val_split)
    num_train_actual = num_train - num_val

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(train_dataset), generator=generator)[:num_train]
    train_indices = indices[:num_train_actual]
    val_indices = indices[num_train_actual:]

    train_subset = Subset(train_dataset, train_indices.tolist())
    val_subset = Subset(train_dataset, val_indices.tolist())

    pin_mem = torch.cuda.is_available()

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_mem,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_mem,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_mem,
    )

    info["num_train"] = num_train_actual
    info["num_val"] = num_val
    info["num_test"] = len(test_dataset)

    return train_loader, val_loader, test_loader, info
