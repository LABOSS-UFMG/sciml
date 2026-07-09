import pytest
import torch

from sciml.utils import initialization


@pytest.mark.parametrize(
    "initializer",
    [
        initialization.xavier_uniform,
        initialization.xavier_normal,
        initialization.kaiming_uniform,
        initialization.kaiming_normal,
        initialization.orthogonal,
    ],
)
def test_initializers_zero_linear_bias_and_keep_weight_finite(initializer):
    layer = torch.nn.Linear(3, 2)
    with torch.no_grad():
        layer.bias.fill_(1.0)

    initializer(layer)

    assert torch.equal(layer.bias, torch.zeros_like(layer.bias))
    assert torch.isfinite(layer.weight).all()


def test_initializers_ignore_non_linear_layers():
    layer = torch.nn.ReLU()

    initialization.xavier_uniform(layer)

    assert isinstance(layer, torch.nn.ReLU)
