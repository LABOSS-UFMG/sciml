# -------------------------------------------------------------------------------- #
import pytest
import torch

from sciml.contracts.context import Context

# -------------------------------------------------------------------------------- #
def test_set_get_contains():
    context = Context()
    x = torch.tensor([[1.0], [2.0]])
    context["x"] = x

    assert "x" in context
    assert "y" not in context
    assert torch.equal(context["x"], x)

def test_requires_grad():
    context = Context()
    context["x"] = torch.tensor([[1.0]])

    context.requires_grad("x")

    assert context["x"].requires_grad

def test_partial_derivative_is_computed_and_cached():
    context = Context()
    x = torch.tensor([[1.0], [2.0], [3.0]], requires_grad=True)
    context["x"] = x
    context["y"] = x ** 2   # y = x^2 -> dy/dx = 2x -> d2y/dx2 = 2

    dy_dx, key = context.partial("y", "x")
    assert torch.allclose(dy_dx, 2 * x)
    assert key in context

    # Second call must return the exact same cached tensor, not recompute it
    dy_dx_again, _ = context.partial("y", "x")
    assert dy_dx_again is dy_dx

    d2y_dx2, _ = context.partial(key, "x")
    assert torch.allclose(d2y_dx2, torch.full_like(x, 2.0))

def test_partial_missing_key_raises():
    context = Context()
    context["x"] = torch.tensor([[1.0]], requires_grad=True)

    with pytest.raises(KeyError):
        context.partial("y", "x")

# -------------------------------------------------------------------------------- #
