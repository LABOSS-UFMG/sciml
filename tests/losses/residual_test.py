import pytest
import torch

from sciml.core.contracts import Batch
from sciml.losses.residual import Residual
from sciml.metrics.mean_squared_error import MeanSquaredError


class ConstantResidual(Residual):
    def residual(self, network, data):
        return network(data["x"]) - data["rhs"]


def test_residual_evaluates_residual_against_zero():
    network = torch.nn.Identity()
    batch = Batch(
        inputs={
            "x": torch.tensor([[1.0], [3.0]]),
            "rhs": torch.tensor([[2.0], [1.0]]),
        }
    )
    loss = ConstantResidual(name="residual", weight=1.0, reduction=MeanSquaredError())
    
    assert torch.isclose(loss.evaluate(network, batch), torch.tensor(2.5))


def test_residual_can_use_default_reduction():
    loss = ConstantResidual(name="residual", weight=1.0)

    assert isinstance(loss.reduction, MeanSquaredError)


def test_residual_is_abstract_without_residual_method():
    with pytest.raises(TypeError):
        Residual(name="residual", weight=1.0, reduction=MeanSquaredError())
