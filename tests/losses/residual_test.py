import pytest
import torch
import warnings

from sciml.core.loss import LossBase
from sciml.core.context import Context
from sciml.core.metric import MetricBase
from sciml.metrics import MeanSquaredError, MeanAbsoluteError
from sciml.losses.residual import Residual


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


# Create a concrete implementation of Residual for testing
class ConcreteResidual(Residual):
    """Concrete implementation of Residual for testing purposes."""
    
    def __init__(self, name: str, weight: float, reduction: MetricBase = MeanSquaredError()):
        super().__init__(name=name, weight=weight, reduction=reduction)
    
    def residual(self, context: Context) -> torch.Tensor:
        """
        Simple residual implementation for testing.
        Returns x^2 - 1, which has zeros at x = ±1.
        """
        x = context["x"]
        return x**2 - 1


def test_residual_init_default():
    """Test Residual initialization with default parameters."""
    loss = ConcreteResidual(name="test_loss", weight=1.0)
    
    assert loss.name == "test_loss"
    assert loss.weight == 1.0
    assert isinstance(loss.reduction, MeanSquaredError)


def test_residual_init_custom_reduction():
    """Test Residual initialization with custom reduction metric."""
    loss = ConcreteResidual(
        name="test_loss", 
        weight=0.5,
        reduction=MeanAbsoluteError()
    )
    
    assert loss.name == "test_loss"
    assert loss.weight == 0.5
    assert isinstance(loss.reduction, MeanAbsoluteError)


def test_residual_init_invalid_name():
    """Test Residual initialization with invalid name."""
    with pytest.raises(TypeError, match="Input must be of type str"):
        ConcreteResidual(name=123, weight=1.0)


def test_residual_init_invalid_weight():
    """Test Residual initialization with invalid weight."""
    with pytest.raises(TypeError, match="Input must be of type float"):
        ConcreteResidual(name="test_loss", weight="1.0")


def test_residual_init_invalid_reduction():
    """Test Residual initialization with invalid reduction."""
    with pytest.raises(TypeError, match="Input must be an instance of"):
        ConcreteResidual(name="test_loss", weight=1.0, reduction="not_a_metric")


def test_residual_evaluate_zero_residual():
    """Test evaluate when residual is zero (perfect solution)."""
    loss = ConcreteResidual(name="test_loss", weight=1.0)
    
    # Create context and store x
    context = Context()
    context["x"] = torch.tensor([1.0, 1.0, 1.0])
    
    result = loss.evaluate(context)
    
    # MSE of zeros should be 0
    assert result == pytest.approx(0.0)
    assert result.shape == torch.Size([])  # Scalar


def test_residual_evaluate_nonzero_residual():
    """Test evaluate when residual is non-zero."""
    loss = ConcreteResidual(name="test_loss", weight=1.0)
    
    # Create context and store x
    context = Context()
    context["x"] = torch.tensor([0.0, 0.0])
    
    result = loss.evaluate(context)
    
    # MSE of [-1, -1] should be 1.0
    assert result == pytest.approx(1.0)
    assert result.shape == torch.Size([])  # Scalar


def test_residual_evaluate_mixed_residuals():
    """Test evaluate with mixed residual values."""
    loss = ConcreteResidual(name="test_loss", weight=1.0)
    
    # Create context and store x
    context = Context()
    context["x"] = torch.tensor([0.0, 1.0, 2.0])
    
    result = loss.evaluate(context)
    
    # MSE of [-1, 0, 3] = (1 + 0 + 9) / 3 = 10/3 ≈ 3.3333
    expected = (1.0 + 0.0 + 9.0) / 3.0
    assert result == pytest.approx(expected)


def test_residual_evaluate_with_different_reduction():
    """Test evaluate with different reduction metric."""
    loss = ConcreteResidual(
        name="test_loss", 
        weight=1.0,
        reduction=MeanAbsoluteError()
    )
    
    # Create context and store x
    context = Context()
    context["x"] = torch.tensor([0.0, 1.0, 2.0])
    
    result = loss.evaluate(context)
    
    # MAE of [-1, 0, 3] = (1 + 0 + 3) / 3 = 4/3 ≈ 1.3333
    expected = (1.0 + 0.0 + 3.0) / 3.0
    assert result == pytest.approx(expected)


def test_residual_evaluate_preserves_gradients():
    """Test that evaluate preserves computational graph for gradients."""
    loss = ConcreteResidual(name="test_loss", weight=1.0)
    
    x = torch.tensor([0.5, 1.5], requires_grad=True)
    context = Context()
    context["x"] = x
    
    result = loss.evaluate(context)
    
    # result should require gradients
    assert result.requires_grad
    
    # Compute gradient
    result.backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape


def test_residual_evaluate_with_batch():
    """Test evaluate with a batch of different points."""
    loss = ConcreteResidual(name="test_loss", weight=1.0)
    
    # Batch of points
    x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    context = Context()
    context["x"] = x
    
    result = loss.evaluate(context)
    
    # Residuals: [3, 0, -1, 0, 3]
    # MSE = (9 + 0 + 1 + 0 + 9) / 5 = 19/5 = 3.8
    expected = 19.0 / 5.0
    assert result == pytest.approx(expected)


def test_residual_inherits_from_lossbase():
    """Test that Residual inherits from LossBase."""
    assert issubclass(Residual, LossBase)
    
    loss = ConcreteResidual(name="test_loss", weight=1.0)
    assert isinstance(loss, LossBase)


def test_residual_abstract_methods():
    """Test that Residual has abstract methods that must be implemented."""
    # Attempting to instantiate Residual directly should raise TypeError
    with pytest.raises(TypeError):
        Residual(name="test", weight=1.0)
    
    # But ConcreteResidual should work
    loss = ConcreteResidual(name="test", weight=1.0)
    assert hasattr(loss, "residual")
    assert callable(loss.residual)


def test_residual_with_complex_context():
    """Test evaluate with a more complex context containing multiple variables."""
    
    class ComplexResidual(Residual):
        def residual(self, context: Context) -> torch.Tensor:
            x = context["x"]
            y = context["y"]
            # Simple residual: x^2 + y^2 - 1 (circle equation)
            return x**2 + y**2 - 1
    
    loss = ComplexResidual(name="circle_loss", weight=1.0)
    
    # Points on the unit circle should have zero residual
    context = Context()
    context["x"] = torch.tensor([0.0, 1.0, 0.0, -1.0])
    context["y"] = torch.tensor([1.0, 0.0, -1.0, 0.0])
    
    result = loss.evaluate(context)
    assert result == pytest.approx(0.0)
    
    # Points off the circle
    context = Context()
    context["x"] = torch.tensor([0.0, 2.0])
    context["y"] = torch.tensor([0.0, 0.0])
    
    result = loss.evaluate(context)
    # Residuals: [-1, 3]
    # MSE = (1 + 9) / 2 = 5
    assert result == pytest.approx(5.0)


def test_residual_evaluate_with_context_containing_derivatives():
    """Test evaluate with context that contains derivative values."""
    
    class DerivativeResidual(Residual):
        def residual(self, context: Context) -> torch.Tensor:
            # Compute derivative if not available
            u_x, _ = context.partial("u", "x", order=1)
            
            # Simple residual: u_x - 2*x (for u = x^2, this should be zero)
            return u_x - 2 * context["x"]
    
    loss = DerivativeResidual(name="derivative_test", weight=1.0)
    
    # Create context with u = x^2
    x = torch.tensor([0.0, 1.0, 2.0], requires_grad=True)
    u = x**2
    
    context = Context()
    context["x"] = x
    context["u"] = u
    
    # Need to set requires_grad on x for derivative computation
    context.requires_grad("x")
    
    result = loss.evaluate(context)
    
    # Residual should be zero everywhere
    assert result.detach().item() == pytest.approx(0.0)