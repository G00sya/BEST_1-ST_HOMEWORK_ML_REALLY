import os
from pathlib import Path
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
import argparse


def prepare_cifar10(data_dir: str = "./data", download: bool = True) -> None:
    """Download and prepare CIFAR-10 dataset with train/test split.

    Args:
        data_dir: Root directory where dataset will be stored
        download: Whether to download the dataset if not exists
    """
    # Create directories if not exist
    train_dir = os.path.join(data_dir, "train")
    test_dir = os.path.join(data_dir, "test")
    Path(train_dir).mkdir(parents=True, exist_ok=True)
    Path(test_dir).mkdir(parents=True, exist_ok=True)

    # Define common transforms
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
        ]
    )

    # Download and save datasets
    CIFAR10(root=train_dir, train=True, transform=transform, download=download)
    CIFAR10(root=test_dir, train=False, transform=transform, download=download)

    print(f"CIFAR-10 dataset prepared at {data_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", default="./data", help="Directory to store dataset"
    )
    parser.add_argument(
        "--no_download",
        action="store_false",
        dest="download",
        help="Skip downloading if dataset exists",
    )
    args = parser.parse_args()

    prepare_cifar10(data_dir=args.data_dir, download=args.download)
