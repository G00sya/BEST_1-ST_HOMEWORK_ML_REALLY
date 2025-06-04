import argparse
from pathlib import Path
import sys
import os

# Добавляем корень проекта в PYTHONPATH
sys.path.append(str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import wandb
from tqdm import tqdm, trange
from src.data.hparams import config


def compute_accuracy(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute accuracy given predictions and targets."""
    return (targets == preds).float().mean().item()


def get_datasets(data_dir: str):
    """Create and return train and test datasets."""
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
        ]
    )

    # Создаем директории, если их нет
    os.makedirs(f"{data_dir}/train", exist_ok=True)
    os.makedirs(f"{data_dir}/test", exist_ok=True)

    train_dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        transform=transform,
        download=True,  # Автоматически скачивает, если нет данных
    )

    test_dataset = datasets.CIFAR10(
        root=data_dir, train=False, transform=transform, download=True
    )

    return train_dataset, test_dataset


def get_model(config: dict) -> nn.Module:
    """Initialize and return the model."""
    model = models.resnet18(
        pretrained=False,
        num_classes=10,
        zero_init_residual=config["zero_init_residual"],
    )
    return model


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train model for one epoch."""
    model.train()
    total_loss = 0.0

    for images, labels in tqdm(train_loader, desc="Training"):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


def evaluate(model, test_loader, device):
    """Evaluate model on test set."""
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            all_preds.append(preds)
            all_labels.append(labels)

    accuracy = compute_accuracy(torch.cat(all_preds), torch.cat(all_labels))
    return accuracy


def main(config: dict, data_dir: str):
    """Main training loop."""
    # Initialize wandb
    wandb.init(config=config, project="effdl_example", name="baseline")

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get data
    train_dataset, test_dataset = get_datasets(data_dir)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=config["batch_size"]
    )

    # Initialize model
    model = get_model(config).to(device)
    wandb.watch(model)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )

    # Training loop
    for epoch in trange(config["epochs"], desc="Epochs"):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        test_acc = evaluate(model, test_loader, device)

        # Log metrics
        wandb.log({"epoch": epoch, "train_loss": train_loss, "test_acc": test_acc})

    # Save model and run ID
    torch.save(model.state_dict(), "model.pt")
    Path("run_id.txt").write_text(wandb.run.id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./CIFAR10",
        help="Directory to store/load dataset",
    )
    args = parser.parse_args()

    main(config, args.data_dir)
