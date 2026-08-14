# -------------------------------------------------------------------------------- #
import torch

from sciml.utils.autograd import derivative

# -------------------------------------------------------------------------------- #
def test_first_order_derivative():
    x = torch.tensor([[1.0], [2.0], [3.0]], requires_grad=True)
    y = x ** 2

    dy_dx = derivative(y, x)

    assert torch.allclose(dy_dx, 2 * x)

def test_second_order_derivative():
    x = torch.tensor([[1.0], [2.0], [3.0]], requires_grad=True)
    y = x ** 3

    dy_dx = derivative(y, x)
    d2y_dx2 = derivative(dy_dx, x)

    assert torch.allclose(d2y_dx2, 6 * x)

# -------------------------------------------------------------------------------- #
