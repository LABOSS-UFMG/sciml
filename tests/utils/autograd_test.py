import torch

from sciml.utils.autograd import derivative


def test_derivative_computes_first_order_gradient():
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    y = x[:, 0:1] ** 2 + 3.0 * x[:, 1:2]

    grad = derivative(y, x)
    expected = torch.tensor([[2.0, 3.0], [6.0, 3.0]])

    assert torch.equal(grad, expected)


def test_derivative_computes_second_order_derivative():
    x = torch.tensor([[1.0], [2.0]], requires_grad=True)
    y = x ** 3

    grad2 = derivative(y, x, order=2)

    assert torch.equal(grad2, 6.0 * x)
