import pytest
import torch

from sciml.core.contracts import Batch
from sciml.losses.supervised import Supervised
from sciml.metrics.mean_absolute_error import MeanAbsoluteError


def test_supervised_evaluates_network_prediction_against_target():
    network = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        network.weight.fill_(2.0)

    batch = Batch(
        inputs={"x": torch.tensor([[1.0], [2.0]])},
        targets={"y": torch.tensor([[3.0], [5.0]])},
    )
    loss = Supervised(name="data", weight=1.0, input_key="x", target_key="y")

    assert torch.isclose(loss.evaluate(network, batch), torch.tensor(1.0))


def test_supervised_accepts_custom_reduction():
    network = torch.nn.Identity()
    batch = Batch(
        inputs={"x": torch.tensor([[1.0], [3.0]])},
        targets={"y": torch.tensor([[2.0], [1.0]])},
    )
    loss = Supervised(
        name="data",
        weight=1.0,
        input_key="x",
        target_key="y",
        reduction=MeanAbsoluteError(),
    )

    assert torch.isclose(loss.evaluate(network, batch), torch.tensor(1.5))


def test_supervised_rejects_invalid_reduction():
    with pytest.raises(TypeError):
        Supervised(
            name="data",
            weight=1.0,
            input_key="x",
            target_key="y",
            reduction=object(),
        )
