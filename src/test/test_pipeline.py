import os

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from unittest.mock import MagicMock, patch
from src.model.train import get_model, train_epoch, evaluate, compute_accuracy, main
from src.data.hparams import config
from src.data.prepare_data import prepare_cifar10
import torchvision.datasets as datasets


@pytest.fixture(scope="module")
def temp_dir():
    test_dir = "./test_data"
    if os.path.exists(test_dir):
        for root, dirs, files in os.walk(test_dir, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        os.rmdir(test_dir)

    yield test_dir

    # Cleanup после всех тестов
    if os.path.exists(test_dir):
        for root, dirs, files in os.walk(test_dir, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        os.rmdir(test_dir)


def test_directories_creation(temp_dir):
    """Проверяем создание директорий"""
    with patch.object(datasets.CIFAR10, "__init__", return_value=None):
        prepare_cifar10(data_dir=temp_dir)

    assert os.path.exists(os.path.join(temp_dir, "train"))
    assert os.path.exists(os.path.join(temp_dir, "test"))


def test_download_called(temp_dir):
    """Проверяем вызов загрузки при download=True"""
    with patch.object(
        datasets.CIFAR10, "__init__", return_value=None
    ) as mock_init, patch.object(datasets.CIFAR10, "download") as mock_download:
        prepare_cifar10(data_dir=temp_dir, download=True)

        # Проверяем что методы были вызваны
        assert mock_init.call_count == 2  # Для train и test
        assert mock_download.call_count == 2  # Проверяем вызов download


def test_no_download(temp_dir):
    """Проверяем пропуск загрузки при download=False"""
    with patch.object(datasets.CIFAR10, "__init__", return_value=None), patch.object(
        datasets.CIFAR10, "download"
    ) as mock_download:
        prepare_cifar10(data_dir=temp_dir, download=False)
        assert not mock_download.called


@pytest.fixture
def train_dataset():
    # Создаем фиктивные данные для тестирования
    images = torch.randn(100, 3, 32, 32)  # 100 примеров, 3 канала, 32x32
    labels = torch.randint(0, 10, (100,))  # 100 меток от 0 до 9
    return TensorDataset(images, labels)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_train_on_one_batch(device, train_dataset):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    device = torch.device(device)

    model = get_model(config).to(device)

    train_loader = DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True
    )
    test_loader = DataLoader(
        train_dataset, batch_size=config["batch_size"]
    )  # Для evaluate

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )

    with torch.no_grad():
        images, labels = next(iter(train_loader))
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)
        accuracy = compute_accuracy(preds, labels)
        assert 0 <= accuracy <= 1

    initial_weights = {name: param.clone() for name, param in model.named_parameters()}

    single_batch_loader = [(next(iter(train_loader)))]  # Создаем loader с одним батчем
    loss = train_epoch(model, single_batch_loader, criterion, optimizer, device)

    assert isinstance(loss, float)
    assert loss > 0
    for name, param in model.named_parameters():
        assert not torch.equal(
            param, initial_weights[name]
        ), f"Weights for {name} did not change!"

    accuracy = evaluate(model, test_loader, device)
    assert 0 <= accuracy <= 1


def test_training(tmp_path):
    """Тест полного цикла обучения с правильным мокингом WandB"""
    test_config = {
        "epochs": 1,
        "batch_size": 8,
        "learning_rate": 0.001,
        "weight_decay": 0,
        "zero_init_residual": True,
    }

    mock_run = MagicMock()
    mock_run.id = "test_run_123"
    mock_run.__enter__.return_value = mock_run

    fake_data = torch.randn(16, 3, 32, 32)
    fake_labels = torch.randint(0, 10, (16,))
    fake_dataset = torch.utils.data.TensorDataset(fake_data, fake_labels)

    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = nn.Linear(3 * 32 * 32, 10)

        def forward(self, x):
            x = x.view(x.size(0), -1)
            return self.layer(x)

    with patch("wandb.init", return_value=mock_run) as mock_init, patch(
        "wandb.watch"
    ) as mock_watch, patch("wandb.log") as mock_log, patch(
        "src.model.train.get_datasets", return_value=(fake_dataset, fake_dataset)
    ), patch("src.model.train.get_model", return_value=TestModel()), patch(
        "src.model.train.Path"
    ) as mock_path, patch("wandb.run", new=mock_run), patch("torch.save") as mock_save:
        mock_file = MagicMock()
        mock_path.return_value = mock_file

        main(test_config, str(tmp_path))

        mock_init.assert_called_once_with(
            config=test_config, project="effdl_example", name="baseline"
        )
        mock_watch.assert_called_once()
        assert mock_log.call_count == 1

        mock_save.assert_called_once()

        mock_path.assert_called_with("run_id.txt")
        mock_file.write_text.assert_called_once_with("test_run_123")
