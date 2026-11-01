"""Data pipeline: CIFAR-10 loading, preprocessing, augmentation and DataLoaders."""
from __future__ import annotations

from typing import List, NamedTuple, Tuple

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from torchvision import transforms as T

from .utils import seed_worker

# Per-channel mean / std of the CIFAR-10 training set.
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
CLASSES = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")


class DataLoaders(NamedTuple):
    train: DataLoader
    val: DataLoader
    test: DataLoader


def build_transforms(augment: bool) -> T.Compose:
    """Normalization for every split; random crop + horizontal flip for training only."""
    normalize = [T.ToTensor(), T.Normalize(CIFAR10_MEAN, CIFAR10_STD)]
    if not augment:
        return T.Compose(normalize)
    return T.Compose([
        T.RandomCrop(32, padding=4, padding_mode="reflect"),
        T.RandomHorizontalFlip(),
        *normalize,
    ])


def split_indices(n: int, val_size: int, seed: int) -> Tuple[List[int], List[int]]:
    """Deterministically split range(n) into (train, validation) indices."""
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=generator).tolist()
    return perm[val_size:], perm[:val_size]


def get_dataloaders(
    data_dir: str = "data",
    batch_size: int = 128,
    val_size: int = 5000,
    num_workers: int = 4,
    seed: int = 42,
    download: bool = True,
) -> DataLoaders:
    """Build train / validation / test loaders.

    The 50k training images are split into 45k for training (augmented) and 5k for
    validation (not augmented). Validation picks the best checkpoint, so the 10k test
    set is only used for the final evaluation.
    """
    # Two views of the same training images: one augmented, one plain.
    train_full = datasets.CIFAR10(data_dir, train=True, transform=build_transforms(True), download=download)
    eval_full = datasets.CIFAR10(data_dir, train=True, transform=build_transforms(False), download=False)
    test_set = datasets.CIFAR10(data_dir, train=False, transform=build_transforms(False), download=download)
    train_idx, val_idx = split_indices(len(train_full), val_size, seed)

    common = dict(
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        worker_init_fn=seed_worker,
    )
    if num_workers > 0:
        common["prefetch_factor"] = 4

    train_loader = DataLoader(
        Subset(train_full, train_idx),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        generator=torch.Generator().manual_seed(seed),
        **common,
    )
    val_loader = DataLoader(Subset(eval_full, val_idx), batch_size=batch_size * 2, shuffle=False, **common)
    test_loader = DataLoader(test_set, batch_size=batch_size * 2, shuffle=False, **common)
    return DataLoaders(train_loader, val_loader, test_loader)


def denormalize(images: torch.Tensor) -> torch.Tensor:
    """Undo normalization so images can be displayed (values clamped to [0, 1])."""
    mean = torch.tensor(CIFAR10_MEAN, device=images.device).view(-1, 1, 1)
    std = torch.tensor(CIFAR10_STD, device=images.device).view(-1, 1, 1)
    return (images * std + mean).clamp(0, 1)
