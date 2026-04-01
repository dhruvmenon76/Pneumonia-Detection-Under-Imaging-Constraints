import os
import copy
from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset_generation import get_dataloaders


# --------------------------------------------------
# Resolution degradation transform
# --------------------------------------------------
class ResolutionDegradation:
    """
    Simulate larger detector pixel size by:
    1. Downsampling the image
    2. Upsampling back to original size

    scale_factor < 1.0 means more degradation
    Example:
        1.0  -> no degradation
        0.85 -> mild degradation
        0.70 -> moderate degradation
        0.50 -> strong degradation
    """
    def __init__(self, output_size=(224, 224), scale_factor=1.0,
                 downsample_mode=Image.Resampling.BILINEAR,
                 upsample_mode=Image.Resampling.BILINEAR):
        self.output_size = output_size
        self.scale_factor = scale_factor
        self.downsample_mode = downsample_mode
        self.upsample_mode = upsample_mode

    def __call__(self, img):
        # force to known size first
        img = img.resize(self.output_size, self.downsample_mode)

        if self.scale_factor == 1.0:
            return img

        w, h = self.output_size
        new_w = max(1, int(round(w * self.scale_factor)))
        new_h = max(1, int(round(h * self.scale_factor)))

        # downsample
        img = img.resize((new_w, new_h), self.downsample_mode)

        # upsample back
        img = img.resize((w, h), self.upsample_mode)

        return img


# --------------------------------------------------
# Build transform pipeline
# --------------------------------------------------
def build_transform(scale_factor, image_size=(224, 224)):
    """
    Builds a transform pipeline that applies
    resolution degradation before tensor conversion.
    """
    return transforms.Compose([
        ResolutionDegradation(output_size=image_size, scale_factor=scale_factor),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])


# --------------------------------------------------
# Create a copy of a dataset with a new transform
# --------------------------------------------------
def clone_dataset_with_transform(dataset, new_transform):
    new_dataset = copy.deepcopy(dataset)
    new_dataset.transform = new_transform
    return new_dataset


# --------------------------------------------------
# Main function
# --------------------------------------------------
def create_resolution_dataloaders(
    base_path,
    batch_size=32,
    degradation_levels=None,
    image_size=(224, 224),
    num_workers=2
):
    """
    Uses your existing dataset_generation.py to build the original loaders,
    then creates new loaders with multiple resolution degradation levels.

    Returns:
        clean_loaders: dict with train/val/test
        degraded_loader_sets: dict keyed by degradation name
    """

    if degradation_levels is None:
        degradation_levels = {
            "clean": 1.00,
            "mild": 0.85,
            "moderate": 0.70,
            "severe": 0.50,
        }

    # ----------------------------------------------
    # Get original loaders from your existing script
    # ----------------------------------------------
    train_loader, val_loader, test_loader = get_dataloaders(
        base_path,
        batch_size=batch_size
    )

    clean_loaders = {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader
    }

    base_train_dataset = train_loader.dataset
    base_val_dataset = val_loader.dataset
    base_test_dataset = test_loader.dataset

    degraded_loader_sets = {}

    for name, scale_factor in degradation_levels.items():
        new_transform = build_transform(scale_factor, image_size=image_size)

        degraded_train_dataset = clone_dataset_with_transform(base_train_dataset, new_transform)
        degraded_val_dataset = clone_dataset_with_transform(base_val_dataset, new_transform)
        degraded_test_dataset = clone_dataset_with_transform(base_test_dataset, new_transform)

        degraded_train_loader = DataLoader(
            degraded_train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers
        )

        degraded_val_loader = DataLoader(
            degraded_val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )

        degraded_test_loader = DataLoader(
            degraded_test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )

        degraded_loader_sets[name] = {
            "train": degraded_train_loader,
            "val": degraded_val_loader,
            "test": degraded_test_loader,
            "scale_factor": scale_factor
        }

    return clean_loaders, degraded_loader_sets


# --------------------------------------------------
# Example usage
# --------------------------------------------------
if __name__ == "__main__":
    base_path = "/content/Pneumonia-Detection-Under-Imaging-Constraints/data"
    batch_size = 32

    degradation_levels = {
        "clean": 1.00,
        "mild": 0.85,
        "moderate": 0.70,
        "severe": 0.50,
    }

    clean_loaders, degraded_loader_sets = create_resolution_dataloaders(
        base_path=base_path,
        batch_size=batch_size,
        degradation_levels=degradation_levels,
        image_size=(224, 224),
        num_workers=2
    )

    print("Clean dataset sizes:")
    print("Train:", len(clean_loaders["train"].dataset))
    print("Val  :", len(clean_loaders["val"].dataset))
    print("Test :", len(clean_loaders["test"].dataset))

    print("\nDegraded datasets created:")
    for name, loader_set in degraded_loader_sets.items():
        print(
            f"{name:10s} | "
            f"scale_factor = {loader_set['scale_factor']:.2f} | "
            f"train size = {len(loader_set['train'].dataset)}"
        )