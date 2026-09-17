"""
src/data.py
Kermany Pediatric Pneumonia dataset loading, preprocessing, and splitting.

Expected dataset structure (Kaggle chest-xray-pneumonia):
    <data_root>/
        train/
            NORMAL/   *.jpeg
            PNEUMONIA/ *.jpeg
        val/
            NORMAL/
            PNEUMONIA/
        test/
            NORMAL/
            PNEUMONIA/

The Kaggle split is pre-made but badly imbalanced (only 16 val images).
We therefore IGNORE the Kaggle split and re-split from the combined pool
using a stratified 70/15/15 train/val/test split.

Usage:
    from src.data import get_dataloaders
    loaders = get_dataloaders(data_root="/path/to/chest_xray", batch_size=32)
    train_loader = loaders["train"]
"""

import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T


# ──────────────────────────────────────────────
# ImageNet normalisation constants (used for ResNet-18 pretrained)
# ──────────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Binary label mapping
LABEL_MAP = {"NORMAL": 0, "PNEUMONIA": 1}


# ──────────────────────────────────────────────
# Transforms
# ──────────────────────────────────────────────
def get_transforms(split: str) -> T.Compose:
    """
    Returns the appropriate torchvision transform pipeline.

    Training: light augmentation (horizontal flip, small rotation).
    Val / Test: deterministic resize + normalize only.
    """
    if split == "train":
        return T.Compose([
            T.Resize((224, 224)),
            T.RandomHorizontalFlip(),
            T.RandomRotation(10),
            T.ColorJitter(brightness=0.2, contrast=0.2),
            T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            T.RandomErasing(p=0.1),
        ])
    else:  # val / test
        return T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])


# ──────────────────────────────────────────────
# Dataset class
# ──────────────────────────────────────────────
class ChestXRayDataset(Dataset):
    """
    Binary chest X-ray dataset (NORMAL=0, PNEUMONIA=1).

    Args:
        image_paths: list of absolute paths to image files
        labels: corresponding integer labels (0 or 1)
        transform: torchvision transform to apply
    """

    def __init__(
        self,
        image_paths: list,
        labels: list,
        transform: Optional[T.Compose] = None,
    ) -> None:
        assert len(image_paths) == len(labels), "paths/labels length mismatch"
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path = self.image_paths[idx]
        label = self.labels[idx]

        img = Image.open(path).convert("RGB")  # chest X-rays are often grayscale; force RGB for ResNet
        if self.transform:
            img = self.transform(img)
        return img, label


# ──────────────────────────────────────────────
# Data loading helpers
# ──────────────────────────────────────────────
def _collect_paths_and_labels(data_root: str) -> Tuple[list, list]:
    """
    Walk the Kaggle chest_xray directory tree and collect all image paths + labels.
    Combines Kaggle's train/val/test splits into a single pool for re-splitting.
    """
    root = Path(data_root)
    paths, labels = [], []

    for subset in ["train", "val", "test"]:
        for class_name, label in LABEL_MAP.items():
            class_dir = root / subset / class_name
            if not class_dir.exists():
                continue
            for ext in ["*.jpeg", "*.jpg", "*.png"]:
                for img_path in class_dir.glob(ext):
                    paths.append(str(img_path))
                    labels.append(label)

    return paths, labels


def get_dataloaders(
    data_root: str,
    batch_size: int = 32,
    num_workers: int = 2,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_seed: int = 42,
) -> Dict[str, DataLoader]:
    """
    Build train/val/test DataLoaders from the Kermany chest X-ray dataset.

    Performs a stratified 70/15/15 split on the combined image pool,
    ignoring Kaggle's original split (which has only 16 val images).

    Args:
        data_root: path to the chest_xray directory (parent of train/val/test)
        batch_size: mini-batch size
        num_workers: DataLoader worker count
        val_size: fraction for validation (default 0.15)
        test_size: fraction for test (default 0.15)
        random_seed: for reproducibility

    Returns:
        tuple: (dict with keys "train", "val", "test" → DataLoader objects,
                class_weights tensor for CrossEntropyLoss)
    """
    paths, labels = _collect_paths_and_labels(data_root)

    if len(paths) == 0:
        raise FileNotFoundError(
            f"No images found under '{data_root}'. "
            "Expected subdirectories: train/NORMAL, train/PNEUMONIA, etc."
        )

    labels_arr = np.array(labels)

    # First split: carve out test set
    train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
        paths, labels,
        test_size=test_size,
        stratify=labels_arr,
        random_state=random_seed,
    )

    # Second split: carve val from train_val
    relative_val_size = val_size / (1.0 - test_size)
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_val_paths, train_val_labels,
        test_size=relative_val_size,
        stratify=train_val_labels,
        random_state=random_seed,
    )

    # Compute class weights for loss function (inverse frequency)
    train_labels_arr = np.array(train_labels)
    class_counts = np.bincount(train_labels_arr)
    class_weights = len(train_labels_arr) / (2.0 * class_counts)
    class_weights_tensor = torch.FloatTensor(class_weights)

    # Compute sample weights for WeightedRandomSampler (to balance batches)
    sample_weights = [1.0 / class_counts[label] for label in train_labels]
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    splits = {
        "train": (train_paths, train_labels),
        "val": (val_paths, val_labels),
        "test": (test_paths, test_labels),
    }

    loaders = {}
    for split_name, (split_paths, split_labels) in splits.items():
        dataset = ChestXRayDataset(
            image_paths=split_paths,
            labels=split_labels,
            transform=get_transforms(split_name),
        )
        
        if split_name == "train":
            loaders[split_name] = DataLoader(
                dataset,
                batch_size=batch_size,
                sampler=sampler,
                num_workers=num_workers,
                pin_memory=torch.cuda.is_available(),
            )
        else:
            loaders[split_name] = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=torch.cuda.is_available(),
            )

    # Print split summary
    print("Dataset split summary:")
    for split_name, loader in loaders.items():
        n = len(loader.dataset)
        n_pneumonia = sum(loader.dataset.labels)
        n_normal = n - n_pneumonia
        print(f"  {split_name:5s}: {n:5d} images  |  Normal: {n_normal}  Pneumonia: {n_pneumonia}")

    return loaders, class_weights_tensor


def get_raw_dataset(
    data_root: str,
    split: str,
    random_seed: int = 42,
) -> ChestXRayDataset:
    """
    Convenience function: returns a single split's Dataset (no DataLoader wrapping).
    Useful for attack generation where you need direct access to image paths/labels.

    Args:
        data_root: path to the chest_xray directory
        split: one of "train", "val", "test"
        random_seed: must match the seed used in get_dataloaders for consistency

    Returns:
        ChestXRayDataset with deterministic eval transforms (no augmentation)
    """
    loaders, _ = get_dataloaders(data_root, batch_size=1, random_seed=random_seed)
    # Return the underlying dataset, but swap to eval transforms (no augmentation)
    dataset = loaders[split].dataset
    dataset.transform = get_transforms("val")  # deterministic, no flip/rotation
    return dataset
