import pytest
import torch

from sciml.utils.initialization import (
    xavier_uniform,
    xavier_normal,
    kaiming_uniform,
    kaiming_normal,
    orthogonal,
)


@pytest.fixture
def linear_layer():
    """Create a linear layer for testing."""
    return torch.nn.Linear(10, 5)


@pytest.fixture
def linear_layer_without_bias():
    """Create a linear layer without bias for testing."""
    return torch.nn.Linear(10, 5, bias=False)


def test_xavier_uniform(linear_layer):
    """Test Xavier uniform initialization on linear layer."""
    # Store initial weights
    initial_weights = linear_layer.weight.clone()
    initial_bias = linear_layer.bias.clone()
    
    # Apply initialization
    xavier_uniform(linear_layer)
    
    # Weights should change
    assert not torch.allclose(linear_layer.weight, initial_weights)
    # Bias should be zeroed
    assert torch.allclose(linear_layer.bias, torch.zeros_like(linear_layer.bias))
    # Check that weights are within expected range for Xavier uniform
    # Xavier uniform is U(-sqrt(6/(fan_in+fan_out)), sqrt(6/(fan_in+fan_out)))
    limit = (6 / (10 + 5)) ** 0.5
    assert torch.all(linear_layer.weight >= -limit)
    assert torch.all(linear_layer.weight <= limit)


def test_xavier_uniform_no_bias(linear_layer_without_bias):
    """Test Xavier uniform initialization on linear layer without bias."""
    # Store initial weights
    initial_weights = linear_layer_without_bias.weight.clone()
    
    # Apply initialization
    xavier_uniform(linear_layer_without_bias)
    
    # Weights should change
    assert not torch.allclose(linear_layer_without_bias.weight, initial_weights)
    # Bias is None, should not raise error
    assert linear_layer_without_bias.bias is None


def test_xavier_normal(linear_layer):
    """Test Xavier normal initialization on linear layer."""
    # Store initial weights
    initial_weights = linear_layer.weight.clone()
    initial_bias = linear_layer.bias.clone()
    
    # Apply initialization
    xavier_normal(linear_layer)
    
    # Weights should change
    assert not torch.allclose(linear_layer.weight, initial_weights)
    # Bias should be zeroed
    assert torch.allclose(linear_layer.bias, torch.zeros_like(linear_layer.bias))
    # Check that weights are from normal distribution (approx mean 0)
    mean = linear_layer.weight.mean().item()
    assert abs(mean) < 0.2  # Should be close to 0


def test_kaiming_uniform_default(linear_layer):
    """Test Kaiming uniform initialization with default nonlinearity (ReLU)."""
    # Store initial weights
    initial_weights = linear_layer.weight.clone()
    initial_bias = linear_layer.bias.clone()
    
    # Apply initialization
    kaiming_uniform(linear_layer)
    
    # Weights should change
    assert not torch.allclose(linear_layer.weight, initial_weights)
    # Bias should be zeroed
    assert torch.allclose(linear_layer.bias, torch.zeros_like(linear_layer.bias))
    # Check that weights are within expected range for Kaiming uniform
    # Kaiming uniform for ReLU: U(-sqrt(6/fan_in), sqrt(6/fan_in))
    limit = (6 / 10) ** 0.5
    assert torch.all(linear_layer.weight >= -limit)
    assert torch.all(linear_layer.weight <= limit)


def test_kaiming_uniform_custom_nonlinearity(linear_layer):
    """Test Kaiming uniform initialization with custom nonlinearity."""
    # Apply with leaky_relu
    kaiming_uniform(linear_layer, nonlinearity='leaky_relu')
    # Just verify it doesn't error and weights are initialized
    assert linear_layer.weight is not None
    
    # Check bias is zeroed
    assert torch.allclose(linear_layer.bias, torch.zeros_like(linear_layer.bias))


def test_kaiming_normal(linear_layer):
    """Test Kaiming normal initialization with default nonlinearity (ReLU)."""
    # Store initial weights
    initial_weights = linear_layer.weight.clone()
    initial_bias = linear_layer.bias.clone()
    
    # Apply initialization
    kaiming_normal(linear_layer)
    
    # Weights should change
    assert not torch.allclose(linear_layer.weight, initial_weights)
    # Bias should be zeroed
    assert torch.allclose(linear_layer.bias, torch.zeros_like(linear_layer.bias))
    # Check that weights are from normal distribution (approx mean 0)
    mean = linear_layer.weight.mean().item()
    assert abs(mean) < 0.2  # Should be close to 0


def test_orthogonal(linear_layer):
    """Test orthogonal initialization on linear layer."""
    # Store initial weights
    initial_weights = linear_layer.weight.clone()
    initial_bias = linear_layer.bias.clone()
    
    # Apply initialization
    orthogonal(linear_layer)
    
    # Weights should change
    assert not torch.allclose(linear_layer.weight, initial_weights)
    # Bias should be zeroed
    assert torch.allclose(linear_layer.bias, torch.zeros_like(linear_layer.bias))
    # Check that rows are orthogonal (Q Q^T = I)
    # For shape (5, 10), rows should be orthogonal
    weight = linear_layer.weight
    # Check orthogonality: W @ W.T should be close to identity
    orthogonal_check = torch.matmul(weight, weight.T)
    identity = torch.eye(weight.shape[0])
    assert torch.allclose(orthogonal_check, identity, atol=1e-6)


def test_initializers_skip_non_linear(linear_layer):
    """Test that initializers skip non-linear layers (pass through)."""
    # Create a non-linear layer (e.g., ReLU)
    non_linear = torch.nn.ReLU()
    
    # All initializers should not error and should not modify non-linear layers
    xavier_uniform(non_linear)
    xavier_normal(non_linear)
    kaiming_uniform(non_linear)
    kaiming_normal(non_linear)
    orthogonal(non_linear)
    
    # ReLU has no parameters, so this is just a pass-through test
    assert True