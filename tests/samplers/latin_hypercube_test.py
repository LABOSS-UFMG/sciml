import pytest
import torch
import numpy as np
import warnings

from sciml.core.sampler import SamplerBase
from sciml.core.context import Context
from sciml.samplers.latin_hypercube import LatinHypercube


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


def test_latin_hypercube_init_default():
    """Test LatinHypercube initialization with default parameters."""
    sampler = LatinHypercube(
        dim=2,
        num_points=100,
        batch_size=10,
        seed=42
    )
    
    assert sampler.dim == 2
    assert sampler.num_points == 100
    assert sampler.batch_size == 10
    assert sampler.input_keys == ["x1", "x2"]
    assert sampler.bounds is None
    assert sampler.insertions is None
    assert sampler.target_keys is None
    assert sampler.target_fn is None
    assert sampler.seed == 42
    assert sampler.device == "cpu"
    assert sampler.dtype == torch.float32
    assert len(sampler._batches) == 10  # 100/10 = 10 batches
    assert sampler._pointer == 0


def test_latin_hypercube_init_custom_keys():
    """Test LatinHypercube initialization with custom input keys."""
    sampler = LatinHypercube(
        dim=2,
        num_points=100,
        batch_size=10,
        input_keys=["x", "t"],
        seed=42
    )
    
    assert sampler.input_keys == ["x", "t"]


def test_latin_hypercube_init_with_bounds():
    """Test LatinHypercube initialization with bounds."""
    sampler = LatinHypercube(
        dim=2,
        num_points=100,
        batch_size=10,
        bounds=[(1, (0.0, 10.0))],
        seed=42
    )
    
    assert sampler.bounds == [(1, (0.0, 10.0))]


def test_latin_hypercube_init_with_insertions():
    """Test LatinHypercube initialization with insertions."""
    sampler = LatinHypercube(
        dim=2,
        num_points=100,
        batch_size=10,
        insertions=[(0, 0.0)],
        seed=42
    )
    
    assert sampler.insertions == [(0, 0.0)]


def test_latin_hypercube_init_with_targets():
    """Test LatinHypercube initialization with target function."""
    def target_fn(inputs):
        return inputs[0] + inputs[1]
    
    sampler = LatinHypercube(
        dim=2,
        num_points=100,
        batch_size=10,
        target_keys=["u"],
        target_fn=target_fn,
        seed=42
    )
    
    assert sampler.target_keys == ["u"]
    assert sampler.target_fn == target_fn


def test_latin_hypercube_init_invalid_dim():
    """Test LatinHypercube initialization with invalid dimension."""
    with pytest.raises(Exception):  # Should raise an error
        LatinHypercube(dim=0, num_points=100, batch_size=10)


def test_latin_hypercube_init_invalid_num_points():
    """Test LatinHypercube initialization with invalid num_points."""
    with pytest.raises(Exception):  # Should raise an error
        LatinHypercube(dim=2, num_points=0, batch_size=10)


def test_latin_hypercube_init_invalid_batch_size():
    """Test LatinHypercube initialization with invalid batch_size."""
    with pytest.raises(Exception):  # Should raise an error
        LatinHypercube(dim=2, num_points=100, batch_size=0)


def test_latin_hypercube_init_batch_size_greater_than_num_points():
    """Test LatinHypercube initialization with batch_size > num_points."""
    with pytest.raises(Exception):  # Should raise an error
        LatinHypercube(dim=2, num_points=10, batch_size=20)


def test_latin_hypercube_next_basic():
    """Test basic next() functionality."""
    sampler = LatinHypercube(
        dim=2,
        num_points=100,
        batch_size=10,
        input_keys=["x", "t"],
        seed=42
    )
    
    context = sampler.next()
    
    assert isinstance(context, Context)
    assert "x" in context
    assert "t" in context
    assert context["x"].shape == torch.Size([10, 1])
    assert context["t"].shape == torch.Size([10, 1])
    assert context["x"].dtype == torch.float32
    assert context["t"].dtype == torch.float32


def test_latin_hypercube_next_with_bounds():
    """Test next() with bounds applied."""
    sampler = LatinHypercube(
        dim=2,
        num_points=100,
        batch_size=10,
        input_keys=["x", "t"],
        bounds=[(0, (0.0, 1.0)), (1, (0.0, 10.0))],
        seed=42
    )
    
    context = sampler.next()
    
    # Check that values are within bounds
    assert torch.all(context["x"] >= 0.0)
    assert torch.all(context["x"] <= 1.0)
    assert torch.all(context["t"] >= 0.0)
    assert torch.all(context["t"] <= 10.0)


def test_latin_hypercube_next_with_insertions():
    """Test next() with insertions (constant dimensions)."""
    sampler = LatinHypercube(
        dim=2,
        num_points=100,
        batch_size=10,
        input_keys=["x", "t"],
        insertions=[(0, 0.0)],
        seed=42
    )
    
    context = sampler.next()
    
    # x should be constant 0.0
    assert torch.all(context["x"] == 0.0)
    # t should be sampled
    assert torch.all(context["t"] != 0.0)  # Likely not all zero


def test_latin_hypercube_next_with_targets():
    """Test next() with target function."""
    def target_fn(inputs):
        # Simple target: sum of inputs
        return inputs[0] + inputs[1]
    
    sampler = LatinHypercube(
        dim=2,
        num_points=100,
        batch_size=10,
        input_keys=["x", "t"],
        target_keys=["u"],
        target_fn=target_fn,
        seed=42
    )
    
    context = sampler.next()
    
    assert "x" in context
    assert "t" in context
    assert "u" in context
    assert context["u"].shape == torch.Size([10, 1])
    
    # Verify target computation
    expected = context["x"] + context["t"]
    assert torch.allclose(context["u"], expected)


def test_latin_hypercube_next_cycles():
    """Test that next() cycles through batches."""
    sampler = LatinHypercube(
        dim=2,
        num_points=10,
        batch_size=5,
    )

    # Get first batch
    context1 = sampler.next()
    assert sampler._pointer == 1
    
    # Get second batch
    context2 = sampler.next()
    assert sampler._pointer == 0

    # They should be different
    # Check if any of the values differ
    diff1 = torch.any(context1["x1"] != context2["x1"])
    diff2 = torch.any(context1["x2"] != context2["x2"])
    assert diff1 or diff2, "Batches should be different"

def test_latin_hypercube_next_cycles_after_last_batch():
    """Test that next() cycles back to first batch after last."""
    sampler = LatinHypercube(
        dim=2,
        num_points=25,
        batch_size=10,
        seed=42
    )
    
    # Number of batches: 25/10 = 3 (last batch has 5 points)
    num_batches = len(sampler._batches)
    
    # Get all batches
    contexts = []
    for _ in range(num_batches):
        contexts.append(sampler.next())
    
    # After last batch, pointer should be 0
    assert sampler._pointer == 0
    
    # Next call should return first batch again
    first_batch = sampler.next()
    assert torch.allclose(first_batch["x1"], contexts[0]["x1"])
    assert torch.allclose(first_batch["x2"], contexts[0]["x2"])


def test_latin_hypercube_last_batch_size():
    """Test that last batch may have fewer points."""
    sampler = LatinHypercube(
        dim=2,
        num_points=25,
        batch_size=10,
        seed=42
    )
    
    # Get all batches
    contexts = []
    for _ in range(len(sampler._batches)):
        contexts.append(sampler.next())
    
    # First two batches should have 10 points
    assert contexts[0]["x1"].shape[0] == 10
    assert contexts[1]["x1"].shape[0] == 10
    # Last batch should have 5 points
    assert contexts[2]["x1"].shape[0] == 5


def test_latin_hypercube_seed_reproducibility():
    """Test that seed produces reproducible results."""
    sampler1 = LatinHypercube(
        dim=2,
        num_points=100,
        batch_size=10,
        seed=42
    )
    
    sampler2 = LatinHypercube(
        dim=2,
        num_points=100,
        batch_size=10,
        seed=42
    )
    
    context1 = sampler1.next()
    context2 = sampler2.next()
    
    # Results should be identical with same seed
    assert torch.allclose(context1["x1"], context2["x1"])
    assert torch.allclose(context1["x2"], context2["x2"])


def test_latin_hypercube_different_seeds():
    """Test that different seeds produce different results."""
    sampler1 = LatinHypercube(
        dim=2,
        num_points=100,
        batch_size=10,
        seed=42
    )
    
    sampler2 = LatinHypercube(
        dim=2,
        num_points=100,
        batch_size=10,
        seed=43
    )
    
    context1 = sampler1.next()
    context2 = sampler2.next()
    
    # Results should be different with different seeds
    assert not torch.allclose(context1["x1"], context2["x1"])


def test_latin_hypercube_dtype_and_device():
    """Test that dtype and device are respected."""
    sampler = LatinHypercube(
        dim=2,
        num_points=100,
        batch_size=10,
        dtype=torch.float64,
        device="cpu",
        seed=42
    )
    
    context = sampler.next()
    
    assert context["x1"].dtype == torch.float64
    assert context["x2"].dtype == torch.float64
    assert str(context["x1"].device) == "cpu"


def test_latin_hypercube_inherits_from_samplerbase():
    """Test that LatinHypercube inherits from SamplerBase."""
    assert issubclass(LatinHypercube, SamplerBase)
    
    sampler = LatinHypercube(dim=2, num_points=100, batch_size=10)
    assert isinstance(sampler, SamplerBase)


def test_latin_hypercube_next_context_contains_all_keys():
    """Test that context contains all specified keys."""
    sampler = LatinHypercube(
        dim=3,
        num_points=10,
        batch_size=5,
        input_keys=["x", "y", "z"],
        target_keys=["u", "v"],
        target_fn=lambda inputs: np.concat([inputs[0] + inputs[1], inputs[1] + inputs[2]], axis=1),
        seed=42
    )
    
    context = sampler.next()
    
    assert "x" in context
    assert "y" in context
    assert "z" in context
    assert "u" in context
    assert "v" in context


def test_latin_hypercube_target_fn_multiple_outputs():
    """Test target function with multiple outputs."""
    def target_fn(inputs):
        x, t = inputs
        return np.concat([x + t, x - t], axis=1)
    
    sampler = LatinHypercube(
        dim=2,
        num_points=100,
        batch_size=10,
        input_keys=["x", "t"],
        target_keys=["u", "v"],
        target_fn=target_fn,
        seed=42
    )
    
    context = sampler.next()
    
    expected_u = context["x"] + context["t"]
    expected_v = context["x"] - context["t"]
    
    assert torch.allclose(context["u"], expected_u)
    assert torch.allclose(context["v"], expected_v)


def test_latin_hypercube_target_fn_numpy():
    """Test that target function works with numpy arrays."""
    def target_fn(inputs):
        # Return numpy arrays
        return np.array(inputs[0] + inputs[1])
    
    sampler = LatinHypercube(
        dim=2,
        num_points=100,
        batch_size=10,
        input_keys=["x", "t"],
        target_keys=["u"],
        target_fn=target_fn,
        seed=42
    )
    
    context = sampler.next()
    
    expected = context["x"] + context["t"]
    assert torch.allclose(context["u"], expected)


def test_latin_hypercube_context_has_expected_shapes():
    """Test that context tensors have expected shapes."""
    sampler = LatinHypercube(
        dim=2,
        num_points=100,
        batch_size=10,
        input_keys=["x", "t"],
        seed=42
    )
    
    context = sampler.next()
    
    # Inputs should have shape (batch_size,)
    assert context["x"].shape == (10, 1)
    assert context["t"].shape == (10, 1)