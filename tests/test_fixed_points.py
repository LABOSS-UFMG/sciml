# -------------------------------------------------------------------------------- #
import pytest
import torch

from sciml.implementations.losses import Supervised
from sciml.implementations.samplers import FixedPoints

# -------------------------------------------------------------------------------- #
def test_boundary_values_and_derivatives():
    boundary = FixedPoints({"x": [0, 1], "u_bc": [1, 2]}, dtype=torch.float64)
    context = boundary.next()
    context.requires_grad("x")
    context["u"] = 1 + context["x"].square()

    loss = Supervised("boundary", ["u"], ["u_bc"])
    assert loss.evaluate(context).item() == 0
    derivative, _ = context.partial("u", "x")
    torch.testing.assert_close(derivative, torch.tensor([[0.], [2.]], dtype=torch.float64))

def test_fixed_points_are_independent_of_input_and_previous_context():
    x = torch.tensor([[1.], [2.]])
    sampler = FixedPoints({"x": x})
    x.zero_()
    first = sampler.next()
    first["x"].add_(10)
    first["u"] = first["x"].square()

    second = sampler.next()
    assert "u" not in second
    torch.testing.assert_close(second["x"], torch.tensor([[1.], [2.]]))

@pytest.mark.parametrize(
    "points",
    [
        {},
        {"x": []},
        {"x": [[1, 2]]},
        {"x": [1, 2], "t": [3]},
    ],
)
def test_invalid_columns_raise(points):
    with pytest.raises(ValueError):
        FixedPoints(points)

# -------------------------------------------------------------------------------- #
