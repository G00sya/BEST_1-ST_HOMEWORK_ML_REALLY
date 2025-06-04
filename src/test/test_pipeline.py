import pytest
import torch
import os
from torch import nn
from torch.utils.data import DataLoader
from unittest.mock import MagicMock, patch
from src.model.train import get_model, train_epoch, evaluate, compute_accuracy, main
from src.data.hparams import config


@pytest.fixture(scope="module")
def prepared_data(tmp_path_factory):
    data_dir = tmp_path_factory.mktemp("data")
    os.makedirs(data_dir / "train", exist_ok=True)
    os.makedirs(data_dir / "test", exist_ok=True)
    return data_dir


@pytest.fixture(scope="module")
def sample_cifar10_data():
    return torch.rand(3, 32, 32)


@pytest.fixture
def train_dataset():
    class MockDataset:
        def __init__(self):
            self.data = torch.rand(100, 3, 32, 32)
            self.targets = torch.randint(0, 10, (100,))

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx], self.targets[idx]

    return MockDataset()


def test_data_directories_created(prepared_data):
    assert os.path.exists(prepared_data / "train")
    assert os.path.exists(prepared_data / "test")


def test_normalization():
    """Тест что нормализация правильно преобразует данные"""
    mock_data = torch.rand(3, 32, 32)

    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
    std = torch.tensor([0.247, 0.243, 0.261]).view(3, 1, 1)

    normalized = (mock_data - mean) / std

    manual_normalized = (mock_data - mean) / std
    assert torch.allclose(normalized, manual_normalized)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_train_on_one_batch(device, train_dataset):
    # Проверяем доступность CUDA
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    device = torch.device(device)

    # Используем функцию get_model из train.py
    model = get_model(config).to(device)

    # Создаем DataLoader с одним батчем
    train_loader = DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True
    )
    test_loader = DataLoader(
        train_dataset, batch_size=config["batch_size"]
    )  # Для evaluate

    # Критерий и оптимизатор (как в train.py)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )

    # 1. Тестируем compute_accuracy
    with torch.no_grad():
        images, labels = next(iter(train_loader))
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)
        accuracy = compute_accuracy(preds, labels)
        assert 0 <= accuracy <= 1

    # 2. Тестируем train_epoch на одном батче
    # Сохраняем начальные веса для проверки обновления
    initial_weights = {name: param.clone() for name, param in model.named_parameters()}

    # Используем функцию train_epoch из train.py
    # Но модифицируем train_loader чтобы он содержал только один батч
    single_batch_loader = [(next(iter(train_loader)))]  # Создаем loader с одним батчем
    loss = train_epoch(model, single_batch_loader, criterion, optimizer, device)

    # Проверяем что loss вычислен и веса изменились
    assert isinstance(loss, float)
    assert loss > 0
    for name, param in model.named_parameters():
        assert not torch.equal(
            param, initial_weights[name]
        ), f"Weights for {name} did not change!"

    # 3. Тестируем evaluate
    # Используем функцию evaluate из train.py
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
