import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class ChestXRayDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        root_dir: path to train / test / val directory
        """
        self.image_paths = []
        self.labels = []
        self.transform = transform

        # class mapping
        label_map = {
            "NORMAL": 0,
            "PNEUMONIA": 1
        }

        # walk through directory
        for class_name in ["NORMAL", "PNEUMONIA"]:
            class_dir = os.path.join(root_dir, class_name)

            for fname in os.listdir(class_dir):
                if fname.lower().endswith((".jpeg", ".jpg", ".png")):
                    self.image_paths.append(os.path.join(class_dir, fname))
                    self.labels.append(label_map[class_name])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)


def get_dataloaders(base_path, batch_size=32, num_workers=2):
    """
    base_path should be:
    Documents/Pneumonia-Detection-Under-Imaging-Constraints/data
    """

    # transforms (good baseline for CNNs)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5])
    ])

    train_dataset = ChestXRayDataset(
        root_dir=os.path.join(base_path, "train"),
        transform=transform
    )

    val_dataset = ChestXRayDataset(
        root_dir=os.path.join(base_path, "val"),
        transform=transform
    )

    test_dataset = ChestXRayDataset(
        root_dir=os.path.join(base_path, "test"),
        transform=transform
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers
    )

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers
    )

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    base_path = "../data"  # since script is in src/

    train_loader, val_loader, test_loader = get_dataloaders(base_path)

    # sanity check
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # check one batch
    images, labels = next(iter(train_loader))
    print(f"Batch shape: {images.shape}")
    print(f"Labels: {labels[:10]}")