import torch
import pytest

from src.model.train import compute_accuracy


def test_empty_input():
    result = compute_accuracy(torch.tensor([]), torch.tensor([]))
    assert torch.isnan(result) or result == 0.0


def test_different_dtypes():
    preds = torch.tensor([1, 0, 1], dtype=torch.float32)
    targets = torch.tensor([1, 0, 0], dtype=torch.int64)
    assert compute_accuracy(preds, targets) == 2 / 3


def test_mismatched_shapes():
    with pytest.raises(RuntimeError):
        compute_accuracy(torch.tensor([1, 2]), torch.tensor([1, 2, 3]))


def test_arange_elems():
    arr = torch.arange(0, 10, dtype=torch.float)
    assert torch.allclose(arr[-1], torch.tensor([9.0]))

    assert torch.allclose(arr[-1], torch.tensor([9], dtype=torch.float))


def test_div_zero_torch():
    a = torch.zeros(1, dtype=torch.float32)
    b = torch.ones(1, dtype=torch.float32)

    result = b / a
    assert torch.isinf(result)


def test_div_zero_python():
    with pytest.raises(ZeroDivisionError):
        1 / 0


def test_accuracy():
    # Best variant
    preds = torch.tensor([1, 2, 3])
    targets = torch.tensor([1, 2, 3])
    assert compute_accuracy(preds, targets) == 1.0

    # worst variant
    preds = torch.tensor([1, 2, 3])
    targets = torch.tensor([4, 5, 6])
    assert compute_accuracy(preds, targets) == 0.0

    # So-so
    preds = torch.tensor([1, 2, 3, 0, 0, 0])
    targets = torch.tensor([1, 2, 3, 4, 5, 6])
    assert torch.allclose(compute_accuracy(preds, targets), torch.tensor(0.5))


@pytest.mark.parametrize(
    "preds,targets,result",
    [
        (torch.tensor([1, 2, 3]), torch.tensor([1, 2, 3]), 1.0),
        (torch.tensor([1, 2, 3]), torch.tensor([0, 0, 0]), 0.0),
        (torch.tensor([1, 2, 3]), torch.tensor([1, 2, 0]), 2 / 3),  # Tut bila byaka
    ],
)
def test_accuracy_parametrized(preds, targets, result):
    assert torch.allclose(
        compute_accuracy(preds, targets), torch.tensor([result]), rtol=0, atol=1e-5
    )
