import pytest
import torch
import warnings

from sciml.core.loss import LossBase
from sciml.core.context import Context
from sciml.core.metric import MetricBase
from sciml.metrics import MeanSquaredError, MeanAbsoluteError
from sciml.losses.supervised import Supervised


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


def test_supervised_init_default():
    """Test Supervised initialization with default parameters."""
    loss = Supervised(
        name="test_loss",
        weight=1.0,
        input_keys=["x"],
        target_keys=["y"]
    )
    
    assert loss.name == "test_loss"
    assert loss.weight == 1.0
    assert loss.input_keys == ["x"]
    assert loss.target_keys == ["y"]
    assert isinstance(loss.reduction, MeanSquaredError)


def test_supervised_init_custom_reduction():
    """Test Supervised initialization with custom reduction metric."""
    loss = Supervised(
        name="test_loss",
        weight=0.5,
        input_keys=["x1", "x2"],
        target_keys=["y1", "y2"],
        reduction=MeanAbsoluteError()
    )
    
    assert loss.name == "test_loss"
    assert loss.weight == 0.5
    assert loss.input_keys == ["x1", "x2"]
    assert loss.target_keys == ["y1", "y2"]
    assert isinstance(loss.reduction, MeanAbsoluteError)


def test_supervised_init_invalid_name():
    """Test Supervised initialization with invalid name."""
    with pytest.raises(TypeError, match="Input must be of type str"):
        Supervised(name=123, weight=1.0, input_keys=["x"], target_keys=["y"])


def test_supervised_init_invalid_weight():
    """Test Supervised initialization with invalid weight."""
    with pytest.raises(TypeError, match="Input must be of type float"):
        Supervised(name="test_loss", weight="1.0", input_keys=["x"], target_keys=["y"])


def test_supervised_init_invalid_input_keys():
    """Test Supervised initialization with invalid input_keys."""
    with pytest.raises(TypeError):
        Supervised(name="test_loss", weight=1.0, input_keys=[1, 2], target_keys=["y"])


def test_supervised_init_invalid_target_keys():
    """Test Supervised initialization with invalid target_keys."""
    with pytest.raises(TypeError):
        Supervised(name="test_loss", weight=1.0, input_keys=["x"], target_keys=[1, 2])


def test_supervised_init_invalid_reduction():
    """Test Supervised initialization with invalid reduction."""
    with pytest.raises(TypeError, match="Input must be an instance of"):
        Supervised(
            name="test_loss",
            weight=1.0,
            input_keys=["x"],
            target_keys=["y"],
            reduction="not_a_metric"
        )


def test_supervised_evaluate_perfect_prediction():
    """Test evaluate when predictions match targets perfectly."""
    loss = Supervised(
        name="test_loss",
        weight=1.0,
        input_keys=["x"],
        target_keys=["y"]
    )
    
    # Create context with predictions and targets that match
    context = Context()
    context["x"] = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    context["y"] = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    
    result = loss.evaluate(context)
    
    # MSE of zeros should be 0
    assert result == pytest.approx(0.0)
    assert result.shape == torch.Size([])  # Scalar


def test_supervised_evaluate_nonzero_error():
    """Test evaluate when predictions differ from targets."""
    loss = Supervised(
        name="test_loss",
        weight=1.0,
        input_keys=["x"],
        target_keys=["y"]
    )
    
    # Create context with predictions and targets that differ
    context = Context()
    context["x"] = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    context["y"] = torch.tensor([[2.0, 3.0], [4.0, 5.0]])
    
    result = loss.evaluate(context)
    
    # MSE of [[1,1], [1,1]] = 1
    assert result.detach().item() == pytest.approx(1.0)


def test_supervised_evaluate_multiple_inputs_targets():
    """Test evaluate with multiple input and target keys."""
    loss = Supervised(
        name="test_loss",
        weight=1.0,
        input_keys=["x1", "x2"],
        target_keys=["y1", "y2"]
    )
    
    # Create context with multiple inputs and targets
    context = Context()
    context["x1"] = torch.tensor([[1.0], [3.0]])
    context["x2"] = torch.tensor([[2.0], [4.0]])
    context["y1"] = torch.tensor([[1.0], [3.0]])
    context["y2"] = torch.tensor([[2.0], [4.0]])
    
    result = loss.evaluate(context)
    
    # Predictions: [[1,2], [3,4]], Targets: [[1,2], [3,4]]
    # MSE should be 0
    assert result.detach().item() == pytest.approx(0.0)


def test_supervised_evaluate_with_different_reduction():
    """Test evaluate with different reduction metric."""
    loss = Supervised(
        name="test_loss",
        weight=1.0,
        input_keys=["x"],
        target_keys=["y"],
        reduction=MeanAbsoluteError()
    )
    
    # Create context with predictions and targets that differ
    context = Context()
    context["x"] = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    context["y"] = torch.tensor([[2.0, 3.0], [4.0, 5.0]])
    
    result = loss.evaluate(context)
    
    # MAE of [[1,1], [1,1]] = 1
    assert result.detach().item() == pytest.approx(1.0)


def test_supervised_evaluate_preserves_gradients():
    """Test that evaluate preserves computational graph for gradients."""
    loss = Supervised(
        name="test_loss",
        weight=1.0,
        input_keys=["x"],
        target_keys=["y"]
    )
    
    # Create context with tensors that require gradients
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    y = torch.tensor([[2.0, 3.0], [4.0, 5.0]])
    
    context = Context()
    context["x"] = x
    context["y"] = y
    
    result = loss.evaluate(context)
    
    # result should require gradients
    assert result.requires_grad
    
    # Compute gradient
    result.backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape


def test_supervised_evaluate_batch():
    """Test evaluate with a batch of different points."""
    loss = Supervised(
        name="test_loss",
        weight=1.0,
        input_keys=["x"],
        target_keys=["y"]
    )
    
    # Batch of points with varying errors
    context = Context()
    context["x"] = torch.tensor([[0.0], [1.0], [2.0], [3.0]])
    context["y"] = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
    
    result = loss.evaluate(context)
    
    # MSE = (1^2 + 1^2 + 1^2 + 1^2) / 4 = 1
    assert result.detach().item() == pytest.approx(1.0)


def test_supervised_evaluate_with_weights():
    """Test evaluate with weighted samples (through context)."""
    loss = Supervised(
        name="test_loss",
        weight=1.0,
        input_keys=["x"],
        target_keys=["y"]
    )
    
    # Create context with samples
    context = Context()
    context["x"] = torch.tensor([[1.0], [2.0], [3.0]])
    context["y"] = torch.tensor([[2.0], [4.0], [6.0]])
    
    result = loss.evaluate(context)
    
    # MSE = (1 + 4 + 9) / 3 = 14/3 ≈ 4.6667
    expected = (1.0 + 4.0 + 9.0) / 3.0
    assert result.detach().item() == pytest.approx(expected)


def test_supervised_inherits_from_lossbase():
    """Test that Supervised inherits from LossBase."""
    assert issubclass(Supervised, LossBase)
    
    loss = Supervised(name="test_loss", weight=1.0, input_keys=["x"], target_keys=["y"])
    assert isinstance(loss, LossBase)


def test_supervised_evaluate_missing_key():
    """Test evaluate when a key is missing from context."""
    loss = Supervised(
        name="test_loss",
        weight=1.0,
        input_keys=["x", "z"],  # 'z' is not in context
        target_keys=["y"]
    )
    
    context = Context()
    context["x"] = torch.tensor([[1.0, 2.0]])
    context["y"] = torch.tensor([[1.0, 2.0]])
    # 'z' is missing
    
    with pytest.raises(KeyError):
        loss.evaluate(context)


def test_supervised_evaluate_single_sample():
    """Test evaluate with a single sample."""
    loss = Supervised(
        name="test_loss",
        weight=1.0,
        input_keys=["x"],
        target_keys=["y"]
    )
    
    context = Context()
    context["x"] = torch.tensor([[1.0, 2.0]])
    context["y"] = torch.tensor([[1.5, 2.5]])
    
    result = loss.evaluate(context)
    
    # MSE of [0.5, 0.5] = (0.25 + 0.25) / 2 = 0.25
    assert result.detach().item() == pytest.approx(0.25)