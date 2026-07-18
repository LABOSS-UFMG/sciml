import pytest
import torch
import numpy as np
import warnings

from sciml.samplers.scaler import Scaler


# Suppress CUDA warnings
@pytest.fixture(autouse=True)
def suppress_cuda_warnings():
    """Suppress CUDA compatibility warnings."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module="torch.cuda")
        yield


@pytest.fixture(autouse=True)
def force_cpu():
    """Force all tensors to be created on CPU to avoid CUDA warnings."""
    original_device = torch.get_default_device()
    torch.set_default_device('cpu')
    yield
    torch.set_default_device(original_device)


def test_scaler_init_1d():
    """Test Scaler initialization with 1D bounds."""
    scaler = Scaler(bounds=[(2.0, 4.0)])
    
    assert scaler.dim == 1
    assert torch.allclose(scaler.lower, torch.tensor([2.0]))
    assert torch.allclose(scaler.upper, torch.tensor([4.0]))
    assert scaler.bounds.shape == (1, 2)


def test_scaler_init_2d():
    """Test Scaler initialization with 2D bounds."""
    scaler = Scaler(bounds=[(0.0, 10.0), (-1.0, 1.0)])
    
    assert scaler.dim == 2
    assert torch.allclose(scaler.lower, torch.tensor([0.0, -1.0]))
    assert torch.allclose(scaler.upper, torch.tensor([10.0, 1.0]))
    assert scaler.bounds.shape == (2, 2)


def test_scaler_init_3d():
    """Test Scaler initialization with 3D bounds."""
    scaler = Scaler(bounds=[(0.0, 1.0), (-5.0, 5.0), (0.0, 100.0)])
    
    assert scaler.dim == 3
    assert torch.allclose(scaler.lower, torch.tensor([0.0, -5.0, 0.0]))
    assert torch.allclose(scaler.upper, torch.tensor([1.0, 5.0, 100.0]))


def test_scaler_init_invalid_bounds_shape():
    """Test Scaler initialization with invalid bounds shape."""
    with pytest.raises(ValueError):
        Scaler(bounds=[(1.0, 2.0, 3.0)])  # Wrong shape


def test_scaler_init_invalid_bounds_order():
    """Test Scaler initialization with lower bound > upper bound."""
    with pytest.raises(ValueError, match="Each upper bound must be greater than the lower bound"):
        Scaler(bounds=[(4.0, 2.0)])  # lower > upper


def test_scaler_init_invalid_bounds_equal():
    """Test Scaler initialization with equal bounds."""
    with pytest.raises(ValueError, match="Each upper bound must be greater than the lower bound"):
        Scaler(bounds=[(2.0, 2.0)])  # lower == upper


def test_scaler_transform_torch_1d():
    """Test transform with torch tensor (1D)."""
    scaler = Scaler(bounds=[(2.0, 4.0)])
    
    x = torch.tensor([[0.0], [0.5], [1.0]])
    result = scaler.transform(x)
    
    expected = torch.tensor([[2.0], [3.0], [4.0]])
    assert torch.allclose(result, expected)
    assert isinstance(result, torch.Tensor)


def test_scaler_transform_torch_2d():
    """Test transform with torch tensor (2D)."""
    scaler = Scaler(bounds=[(0.0, 10.0), (-1.0, 1.0)])
    
    x = torch.tensor([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
    result = scaler.transform(x)
    
    expected = torch.tensor([[0.0, -1.0], [5.0, 0.0], [10.0, 1.0]])
    assert torch.allclose(result, expected)
    assert isinstance(result, torch.Tensor)


def test_scaler_transform_torch_batch():
    """Test transform with torch tensor (batch)."""
    scaler = Scaler(bounds=[(0.0, 10.0), (-1.0, 1.0), (0.0, 100.0)])
    
    x = torch.tensor([
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.5],
        [1.0, 1.0, 1.0]
    ])
    result = scaler.transform(x)
    
    expected = torch.tensor([
        [0.0, -1.0, 0.0],
        [5.0, 0.0, 50.0],
        [10.0, 1.0, 100.0]
    ])
    assert torch.allclose(result, expected)


def test_scaler_transform_numpy_1d():
    """Test transform with numpy array (1D)."""
    scaler = Scaler(bounds=[(2.0, 4.0)])
    
    x = np.array([[0.0], [0.5], [1.0]], dtype=np.float32)
    result = scaler.transform(x)
    
    expected = np.array([[2.0], [3.0], [4.0]], dtype=np.float32)
    np.testing.assert_allclose(result, expected)
    assert isinstance(result, np.ndarray)


def test_scaler_transform_numpy_2d():
    """Test transform with numpy array (2D)."""
    scaler = Scaler(bounds=[(0.0, 10.0), (-1.0, 1.0)])
    
    x = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]], dtype=np.float32)
    result = scaler.transform(x)
    
    expected = np.array([[0.0, -1.0], [5.0, 0.0], [10.0, 1.0]], dtype=np.float32)
    np.testing.assert_allclose(result, expected)
    assert isinstance(result, np.ndarray)


def test_scaler_transform_numpy_batch():
    """Test transform with numpy array (batch)."""
    scaler = Scaler(bounds=[(0.0, 10.0), (-1.0, 1.0), (0.0, 100.0)])
    
    x = np.array([
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.5],
        [1.0, 1.0, 1.0]
    ], dtype=np.float32)
    result = scaler.transform(x)
    
    expected = np.array([
        [0.0, -1.0, 0.0],
        [5.0, 0.0, 50.0],
        [10.0, 1.0, 100.0]
    ], dtype=np.float32)
    np.testing.assert_allclose(result, expected)


def test_scaler_transform_torch_preserves_dtype():
    """Test that transform preserves torch dtype."""
    scaler = Scaler(bounds=[(0.0, 10.0), (-1.0, 1.0)])
    
    # Test with float32
    x_f32 = torch.tensor([[0.5, 0.5]], dtype=torch.float32)
    result_f32 = scaler.transform(x_f32)
    assert result_f32.dtype == torch.float32
    
    # Test with float64
    x_f64 = torch.tensor([[0.5, 0.5]], dtype=torch.float64)
    result_f64 = scaler.transform(x_f64)
    assert result_f64.dtype == torch.float64


def test_scaler_transform_numpy_preserves_dtype():
    """Test that transform preserves numpy dtype."""
    scaler = Scaler(bounds=[(0.0, 10.0), (-1.0, 1.0)])
    
    # Test with float32
    x_f32 = np.array([[0.5, 0.5]], dtype=np.float32)
    result_f32 = scaler.transform(x_f32)
    assert result_f32.dtype == np.float32
    
    # Test with float64
    x_f64 = np.array([[0.5, 0.5]], dtype=np.float64)
    result_f64 = scaler.transform(x_f64)
    assert result_f64.dtype == np.float64


def test_scaler_transform_torch_invalid_shape():
    """Test transform with torch tensor of invalid shape."""
    scaler = Scaler(bounds=[(0.0, 10.0), (-1.0, 1.0)])
    
    x = torch.tensor([[0.0, 0.0, 0.0]])  # 3 features, but scaler has dim=2
    
    with pytest.raises(ValueError, match="The last dimension of x must be 2"):
        scaler.transform(x)


def test_scaler_transform_numpy_invalid_shape():
    """Test transform with numpy array of invalid shape."""
    scaler = Scaler(bounds=[(0.0, 10.0), (-1.0, 1.0)])
    
    x = np.array([[0.0, 0.0, 0.0]])  # 3 features, but scaler has dim=2
    
    with pytest.raises(ValueError, match="The last dimension of x must be 2"):
        scaler.transform(x)


def test_scaler_transform_invalid_type():
    """Test transform with invalid input type."""
    scaler = Scaler(bounds=[(0.0, 10.0)])
    
    with pytest.raises(TypeError, match="x must be either a torch.Tensor or a numpy.ndarray"):
        scaler.transform([0.0, 0.5, 1.0])  # List is not supported


def test_scaler_transform_torch_edge_cases():
    """Test transform with edge cases (0 and 1 values)."""
    scaler = Scaler(bounds=[(2.0, 4.0), (-1.0, 1.0)])
    
    # Test with 0 values (should map to lower bounds)
    x_zero = torch.tensor([[0.0, 0.0]])
    result_zero = scaler.transform(x_zero)
    expected_zero = torch.tensor([[2.0, -1.0]])
    assert torch.allclose(result_zero, expected_zero)
    
    # Test with 1 values (should map to upper bounds)
    x_one = torch.tensor([[1.0, 1.0]])
    result_one = scaler.transform(x_one)
    expected_one = torch.tensor([[4.0, 1.0]])
    assert torch.allclose(result_one, expected_one)


def test_scaler_transform_torch_negative_bounds():
    """Test transform with negative bounds."""
    scaler = Scaler(bounds=[(-5.0, -2.0), (-10.0, 10.0)])
    
    x = torch.tensor([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
    result = scaler.transform(x)
    
    expected = torch.tensor([[-5.0, -10.0], [-3.5, 0.0], [-2.0, 10.0]])
    assert torch.allclose(result, expected)


def test_scaler_transform_torch_large_bounds():
    """Test transform with large bounds."""
    scaler = Scaler(bounds=[(0.0, 1000000.0), (-1000000.0, 1000000.0)])
    
    x = torch.tensor([[0.5, 0.5]])
    result = scaler.transform(x)
    
    expected = torch.tensor([[500000.0, 0.0]])
    assert torch.allclose(result, expected)


def test_scaler_transform_torch_broadcasting():
    """Test that transform correctly broadcasts to batch dimensions."""
    scaler = Scaler(bounds=[(0.0, 10.0), (-1.0, 1.0), (0.0, 100.0)])
    
    # Shape: (2, 3) - batch of 2, each with 3 features
    x = torch.tensor([
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0]
    ])
    result = scaler.transform(x)
    
    expected = torch.tensor([
        [0.0, -1.0, 0.0],
        [10.0, 1.0, 100.0]
    ])
    assert torch.allclose(result, expected)
    assert result.shape == (2, 3)


def test_scaler_transform_torch_1d_single_sample():
    """Test transform with 1D single sample."""
    scaler = Scaler(bounds=[(0.0, 10.0), (-1.0, 1.0)])
    
    x = torch.tensor([0.5, 0.5])  # 1D tensor
    result = scaler.transform(x)
    
    expected = torch.tensor([5.0, 0.0])
    assert torch.allclose(result, expected)
    assert result.shape == (2,)


def test_scaler_transform_numpy_1d_single_sample():
    """Test transform with numpy 1D single sample."""
    scaler = Scaler(bounds=[(0.0, 10.0), (-1.0, 1.0)])
    
    x = np.array([0.5, 0.5])
    result = scaler.transform(x)
    
    expected = np.array([5.0, 0.0])
    np.testing.assert_allclose(result, expected)
    assert result.shape == (2,)