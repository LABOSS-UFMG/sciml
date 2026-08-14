# -------------------------------------------------------------------------------- #
import pytest
import torch

from sciml.contracts.context import Context
from sciml.implementations.losses import Residual, Supervised

# -------------------------------------------------------------------------------- #
def test_residual_zero_when_pde_satisfied():
    context = Context()
    loss = Residual(name="r", weight=1.0, residual=lambda ctx: torch.zeros(4, 1))

    assert loss.evaluate(context).item() == pytest.approx(0.0)

def test_residual_uses_mse_reduction_by_default():
    context = Context()
    loss = Residual(name="r", weight=1.0, residual=lambda ctx: torch.tensor([[1.0], [-1.0]]))
    # residual is compared against zero via the default MeanSquaredError reduction

    assert loss.evaluate(context).item() == pytest.approx(1.0)

def test_supervised_zero_when_matching():
    context = Context()
    context["y"] = torch.tensor([[1.0], [2.0]])
    context["target"] = torch.tensor([[1.0], [2.0]])

    loss = Supervised(name="s", input_keys=["y"], target_keys=["target"])

    assert loss.evaluate(context).item() == pytest.approx(0.0)

def test_supervised_nonzero_when_mismatched():
    context = Context()
    context["y"] = torch.tensor([[1.0]])
    context["target"] = torch.tensor([[3.0]])

    loss = Supervised(name="s", input_keys=["y"], target_keys=["target"])

    assert loss.evaluate(context).item() == pytest.approx(4.0)

# -------------------------------------------------------------------------------- #
