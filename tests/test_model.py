# -------------------------------------------------------------------------------- #
import pytest
import torch

from conftest import LinearModel

# -------------------------------------------------------------------------------- #
@pytest.mark.parametrize("training", [True, False])
def test_evaluation_mode_preserves_input_gradients(training):
    model = LinearModel()
    model.network.train(training)
    x = torch.ones(1, 1, requires_grad=True)
    with model.evaluation_mode():
        assert model.network.training == training
        (derivative,) = torch.autograd.grad(model.network(x).sum(), x)
        torch.testing.assert_close(derivative, model.network.weight)
    assert model.network.training == training

def test_evaluation_mode_propagates_exceptions():
    model = LinearModel()
    with pytest.raises(RuntimeError, match="validation failed"):
        with model.evaluation_mode():
            raise RuntimeError("validation failed")

# -------------------------------------------------------------------------------- #
