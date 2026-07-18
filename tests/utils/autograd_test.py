import pytest
import torch

from sciml.utils.autograd import derivative

def test_derivative_orders():
    """Test first, second, and third order derivatives."""
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    y = (x[:, 0]**2 + x[:, 1]**2).reshape(-1, 1)
    
    # First derivative: [2*x1, 2*x2]
    first = derivative(y, x, order=1)
    expected_first = torch.tensor([[2.0, 4.0], [6.0, 8.0]])
    torch.testing.assert_close(first, expected_first)
    
    # Second derivative: [2, 2]
    second = derivative(y, x, order=2)
    expected_second = torch.tensor([[2.0, 2.0], [2.0, 2.0]])
    torch.testing.assert_close(second, expected_second)
    
    # Third derivative: [0, 0]
    third = derivative(y, x, order=3)
    expected_third = torch.zeros_like(x)
    torch.testing.assert_close(third, expected_third)


def test_derivative_linear_function():
    """Test first derivative of linear function."""
    x = torch.tensor([[1.0, 2.0]], requires_grad=True)
    y = (2 * x[:, 0] + 3 * x[:, 1]).reshape(-1, 1)
    
    result = derivative(y, x, order=1)
    expected = torch.tensor([[2.0, 3.0]])
    torch.testing.assert_close(result, expected)


def test_derivative_graph_preserved():
    """Verify computational graph is preserved for higher-order derivatives."""
    x = torch.tensor([[1.0, 2.0]], requires_grad=True)
    y = (x[:, 0]**2 + x[:, 1]**2).reshape(-1, 1)
    
    first_deriv = derivative(y, x, order=1)
    
    # Compute gradient of first derivative to verify graph preservation
    grad = torch.autograd.grad(first_deriv.sum(), x, create_graph=False)[0]
    expected = torch.tensor([[2.0, 2.0]])
    torch.testing.assert_close(grad, expected)
