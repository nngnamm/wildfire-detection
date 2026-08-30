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
from PIL import ImageFile
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Large datasets pulled from a zip (e.g. downloaded from Kaggle) sometimes
# contain a handful of images that got slightly cut off during download or
# extraction. By default, PIL refuses to load a truncated image and raises
# an OSError, which would otherwise crash an entire multi-hour training
# run over one bad file deep into the dataset. Setting this flag tells PIL
# to load as much of the image data as it can instead of raising. It's a
# safety net, not a substitute for actually finding and reviewing broken
# files — see verify_dataset() below.
ImageFile.LOAD_TRUNCATED_IMAGES = True

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


def verify_dataset(data_dir: str, remove_corrupted: bool = False):
    """
    Scans every image file under data_dir/{train,valid,test} and tries to
    fully load and decode it. Reports any file that fails, since those are
    the files that will otherwise crash a DataLoader worker mid-training
    (see the OSError: "image file is truncated" case this is meant to
    catch).

    This is a diagnostic utility, not something main.py calls automatically
    on every run — for a 30,000+ image dataset it can take a few minutes,
    so run it once after downloading/extracting the dataset, or whenever
    training crashes with an image-loading error.

    Args:
        data_dir: path to the root `dataset/` folder.
        remove_corrupted: if True, delete any file that fails to load.
                           If False (default), only report — nothing is
                           deleted, so you can inspect the files yourself
                           first.

    Returns:
        List of file paths that failed to load.
    """
    from PIL import Image

    bad_files = []
    splits = ["train", "valid", "test"]
    checked = 0

    for split in splits:
        split_dir = os.path.join(data_dir, split)
        if not os.path.isdir(split_dir):
            continue

        for class_name in sorted(os.listdir(split_dir)):
            class_dir = os.path.join(split_dir, class_name)
            if not os.path.isdir(class_dir):
                continue

            for filename in os.listdir(class_dir):
                file_path = os.path.join(class_dir, filename)
                checked += 1
                try:
                    with Image.open(file_path) as img:
                        img.convert("RGB").load()
                except Exception as e:
                    bad_files.append(file_path)
                    print(f"[verify_dataset] CORRUPTED: {file_path} ({e})")

    print(f"\n[verify_dataset] Checked {checked} files, found {len(bad_files)} corrupted.")

    if bad_files and remove_corrupted:
        for file_path in bad_files:
            os.remove(file_path)
        print(f"[verify_dataset] Removed {len(bad_files)} corrupted file(s).")
    elif bad_files:
        print("[verify_dataset] Files were NOT deleted (remove_corrupted=False). "
              "Re-run with remove_corrupted=True to delete them, or inspect/replace manually.")

    return bad_files


if __name__ == "__main__":
    # Convenience: `python -m src.dataset` (or `python src/dataset.py` from
    # the project root) scans the dataset for corrupted images without
    # running any training. Pass --remove to also delete bad files.
    import argparse

    parser = argparse.ArgumentParser(description="Scan the dataset for corrupted image files.")
    parser.add_argument("--data_dir", type=str, default="dataset")
    parser.add_argument("--remove", action="store_true",
                         help="Delete corrupted files instead of only reporting them.")
    args = parser.parse_args()

    verify_dataset(args.data_dir, remove_corrupted=args.remove)
