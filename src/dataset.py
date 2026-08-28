"""
dataset.py
-----------
Handles data loading, transforms, and DataLoader creation for the
wildfire detection dataset.

Expected directory layout under `dataset/` (this matches the popular
Kaggle "Wildfire Prediction Dataset" layout):

    dataset/
        train/
            nowildfire/
            wildfire/
        valid/
            nowildfire/
            wildfire/
        test/
            nowildfire/
            wildfire/

Because torchvision.datasets.ImageFolder assigns numeric labels in
alphabetical order of the subfolder names, "nowildfire" -> 0 and
"wildfire" -> 1. This is confirmed at load time and printed to the
console so there is never ambiguity about which class is positive.
"""

import os
from typing import Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ImageNet statistics used to normalize inputs, since we are using a
# ResNet50 backbone pre-trained on ImageNet.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGE_SIZE = 224


def get_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Builds the train and eval transform pipelines.

    Training transforms include light augmentation (random horizontal
    flip + random rotation) so the model doesn't overfit to exact pixel
    positions of smoke/fire regions. Validation/test transforms only
    resize + normalize, since we want a stable, reproducible evaluation.
    """
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    return train_transform, eval_transform


def get_dataloaders(data_dir: str, batch_size: int = 32, num_workers: int = 2):
    """
    Creates train/val/test DataLoaders from an ImageFolder-style
    directory structure.

    Args:
        data_dir: path to the root `dataset/` folder containing
                  train/, valid/, and test/ subfolders.
        batch_size: batch size for all three loaders.
        num_workers: number of subprocesses used for data loading.

    Returns:
        train_loader, val_loader, test_loader, class_to_idx (dict)
    """
    train_transform, eval_transform = get_transforms()

    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "valid")
    test_dir = os.path.join(data_dir, "test")

    for path in (train_dir, val_dir, test_dir):
        if not os.path.isdir(path):
            raise FileNotFoundError(
                f"Expected dataset subfolder not found: {path}\n"
                f"Please make sure the Kaggle dataset is extracted into "
                f"'{data_dir}' with train/valid/test subfolders."
            )

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=eval_transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=eval_transform)

    # Sanity check: all splits must agree on the class-to-index mapping.
    assert train_dataset.class_to_idx == val_dataset.class_to_idx == test_dataset.class_to_idx, (
        "Class-to-index mapping mismatch between train/valid/test splits. "
        "Check that each split has identically named subfolders."
    )

    print(f"[dataset] class_to_idx mapping: {train_dataset.class_to_idx}")
    print(f"[dataset] train={len(train_dataset)} | val={len(val_dataset)} | test={len(test_dataset)}")

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, test_loader, train_dataset.class_to_idx
