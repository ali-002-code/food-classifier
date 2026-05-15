"""
src/dataset.py
--------------
Data loading and augmentation for Food-101.

Food-101 facts (useful to know for interviews):
  - 101 food categories (pizza, sushi, ramen, etc.)
  - 750 training images + 250 test images per class
  - 75,750 training images total, 25,250 test images
  - Images are real-world, noisy, and variable aspect ratios
  - Available directly via torchvision.datasets.Food101

Augmentation strategy
---------------------
Training transforms apply *random* perturbations to artificially expand the
effective dataset and prevent the model memorising exact pixel patterns.

The key principle: augmentations should simulate realistic variation in how
a dish photo might be taken — different lighting, slight angle changes, crops
— without producing implausible images (e.g. don't flip vertically, because
upside-down food images don't appear in the wild).

Validation/test transforms are deterministic: only resize and normalise.
You never augment validation data because you want a stable metric.

Normalisation values
--------------------
mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225] are the
ImageNet channel statistics. Even though we're training from scratch,
these are reasonable priors for natural images and are universally used.
They convert pixel values from [0, 1] to approximately [-2, 2], which
keeps activations in the linear region of ReLU initially.
"""

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from typing import Tuple
import numpy as np


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def get_transforms(image_size: int, train: bool) -> transforms.Compose:
    """
    Return the appropriate transform pipeline for train or val/test.

    Args:
        image_size: target square size after resizing
        train: if True, apply random augmentation; if False, deterministic only
    """
    if train:
        return transforms.Compose([
            # Resize slightly larger than target so RandomCrop has room to crop
            transforms.Resize(int(image_size * 1.15)),

            # Random crop to target size: simulates framing variation
            transforms.RandomCrop(image_size),

            # Horizontal flip: a dish looks the same mirrored
            transforms.RandomHorizontalFlip(p=0.5),

            # Colour jitter: simulates different lighting conditions
            # brightness/contrast/saturation each varied by up to 20%
            # hue varied by up to 5% (small — avoid unnatural colours)
            transforms.ColorJitter(brightness=0.2, contrast=0.2,
                                   saturation=0.2, hue=0.05),

            # Small random rotation: simulates slightly tilted camera angle
            transforms.RandomRotation(degrees=15),

            # Convert PIL image to tensor and scale to [0, 1]
            transforms.ToTensor(),

            # Normalise channels to zero-mean unit-variance
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    else:
        return transforms.Compose([
            # For val/test: deterministic centre crop only
            transforms.Resize(int(image_size * 1.15)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])


def get_dataloaders(
    data_dir: str,
    image_size: int,
    batch_size: int,
    num_workers: int,
    subset_fraction: float = 1.0,
) -> Tuple[DataLoader, DataLoader, list]:
    """
    Download Food-101 and return train/test DataLoaders.

    Args:
        data_dir: directory where Food-101 will be downloaded
        image_size: square target resolution
        batch_size: samples per batch
        num_workers: parallel workers for data loading
        subset_fraction: use only this fraction of data (0.0, 1.0].
                         Set to e.g. 0.2 for fast debugging runs.

    Returns:
        train_loader, test_loader, class_names
    """
    train_dataset = datasets.Food101(
        root=data_dir,
        split="train",
        transform=get_transforms(image_size, train=True),
        download=True,
    )

    test_dataset = datasets.Food101(
        root=data_dir,
        split="test",
        transform=get_transforms(image_size, train=False),
        download=True,
    )

    # Optional: use a subset for fast iteration during development
    if subset_fraction < 1.0:
        train_dataset = _subsample(train_dataset, subset_fraction, seed=42)
        test_dataset  = _subsample(test_dataset,  subset_fraction, seed=42)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,           # shuffle every epoch to prevent ordering bias
        num_workers=num_workers,
        pin_memory=True,        # speeds up CPU->GPU transfer when using CUDA
        drop_last=True,         # drop final incomplete batch to keep BatchNorm stable
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,          # never shuffle val/test — reproducible metrics
        num_workers=num_workers,
        pin_memory=True,
    )

    class_names = train_dataset.classes if hasattr(train_dataset, "classes") \
        else train_dataset.dataset.classes

    print(f"Training samples:   {len(train_dataset):,}")
    print(f"Test samples:       {len(test_dataset):,}")
    print(f"Number of classes:  {len(class_names)}")

    return train_loader, test_loader, class_names


def _subsample(dataset, fraction: float, seed: int) -> Subset:
    """Return a stratified subset of a torchvision dataset."""
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(dataset), size=int(len(dataset) * fraction), replace=False)
    return Subset(dataset, indices.tolist())


def denormalise(tensor: torch.Tensor) -> torch.Tensor:
    """
    Reverse ImageNet normalisation for visualisation purposes.
    Input: (C, H, W) or (B, C, H, W) normalised tensor
    Output: same shape with values in [0, 1]
    """
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std  = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (tensor * std + mean).clamp(0, 1)
