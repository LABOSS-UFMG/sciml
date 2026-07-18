import warnings

import pytest
import torch
import tempfile
from pathlib import Path

from sciml.utils.serialization import (
    save_network,
    load_network,
    save_optimizer,
    load_optimizer,
    save_checkpoint,
    load_checkpoint,
    _ensure_parent_dir,
    _process_networks,
    _process_optimizers,
)


# Filter out CUDA warnings globally
@pytest.fixture(autouse=True)
def suppress_cuda_warnings():
    """Suppress CUDA compatibility warnings."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module="torch.cuda")
        yield


@pytest.fixture(autouse=True)
def force_cpu():
    """Force all tensors to be created on CPU to avoid CUDA warnings."""
    # Save original device
    original_device = torch.get_default_device()
    # Set to CPU
    torch.set_default_device('cpu')
    yield
    # Restore original device
    torch.set_default_device(original_device)


@pytest.fixture
def simple_network():
    """Create a simple network for testing."""
    return torch.nn.Sequential(
        torch.nn.Linear(10, 5),
        torch.nn.ReLU(),
        torch.nn.Linear(5, 1)
    )


@pytest.fixture
def simple_optimizer(simple_network):
    """Create a simple optimizer for testing."""
    return torch.optim.Adam(simple_network.parameters(), lr=0.001)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


def test_ensure_parent_dir(temp_dir):
    """Test that parent directory is created if it doesn't exist."""
    # Create nested path
    nested_path = temp_dir / "subdir" / "nested" / "file.pt"
    
    # Parent should not exist yet
    assert not nested_path.parent.exists()
    
    # Call function
    _ensure_parent_dir(nested_path)
    
    # Parent should now exist
    assert nested_path.parent.exists()
    
    # Parent should be a directory
    assert nested_path.parent.is_dir()


def test_ensure_parent_dir_current_or_root():
    """Test ensure_parent_dir with current or root directory (should not create)."""
    # Current directory
    _ensure_parent_dir("file.pt")  # Should not error
    
    # Root directory (Unix-like)
    _ensure_parent_dir("/file.pt")  # Should not error


def test_process_networks_single(simple_network):
    """Test _process_networks with a single network."""
    result = _process_networks(simple_network)
    
    assert isinstance(result, dict)
    assert len(result) == 1
    assert "network" in result
    assert result["network"] is simple_network


def test_process_networks_sequence(simple_network):
    """Test _process_networks with a sequence of networks."""
    net1 = simple_network
    net2 = torch.nn.Linear(10, 5)
    
    result = _process_networks([net1, net2])
    
    assert isinstance(result, dict)
    assert len(result) == 2
    assert "network_0" in result
    assert "network_1" in result
    assert result["network_0"] is net1
    assert result["network_1"] is net2


def test_process_networks_dict(simple_network):
    """Test _process_networks with a dictionary of networks."""
    net1 = simple_network
    net2 = torch.nn.Linear(10, 5)
    
    result = _process_networks({"encoder": net1, "decoder": net2})
    
    assert isinstance(result, dict)
    assert len(result) == 2
    assert "encoder" in result
    assert "decoder" in result
    assert result["encoder"] is net1
    assert result["decoder"] is net2


def test_process_networks_invalid():
    """Test _process_networks with invalid input."""
    with pytest.raises(TypeError, match="network must be a torch.nn.Module, sequence, or dict"):
        _process_networks("not a network")


def test_process_networks_dict_invalid_keys():
    """Test _process_networks with dictionary containing non-string keys."""
    # This should fail because the dict contains non-string keys
    # The error might come from is_dtype or from the key validation
    with pytest.raises(TypeError):
        _process_networks({1: torch.nn.Linear(10, 5)})


def test_process_optimizers_single(simple_optimizer):
    """Test _process_optimizers with a single optimizer."""
    result = _process_optimizers(simple_optimizer)
    
    assert isinstance(result, dict)
    assert len(result) == 1
    assert "optimizer" in result
    assert result["optimizer"] is simple_optimizer


def test_process_optimizers_sequence(simple_network):
    """Test _process_optimizers with a sequence of optimizers."""
    opt1 = torch.optim.Adam(simple_network.parameters(), lr=0.001)
    opt2 = torch.optim.SGD(simple_network.parameters(), lr=0.01)
    
    result = _process_optimizers([opt1, opt2])
    
    assert isinstance(result, dict)
    assert len(result) == 2
    assert "optimizer_0" in result
    assert "optimizer_1" in result
    assert result["optimizer_0"] is opt1
    assert result["optimizer_1"] is opt2


def test_process_optimizers_dict(simple_network):
    """Test _process_optimizers with a dictionary of optimizers."""
    opt1 = torch.optim.Adam(simple_network.parameters(), lr=0.001)
    opt2 = torch.optim.SGD(simple_network.parameters(), lr=0.01)
    
    result = _process_optimizers({"adam": opt1, "sgd": opt2})
    
    assert isinstance(result, dict)
    assert len(result) == 2
    assert "adam" in result
    assert "sgd" in result
    assert result["adam"] is opt1
    assert result["sgd"] is opt2


def test_process_optimizers_invalid():
    """Test _process_optimizers with invalid input."""
    with pytest.raises(TypeError, match="optimizer must be a torch.optim.Optimizer, sequence, or dict"):
        _process_optimizers("not an optimizer")


def test_save_load_network(temp_dir, simple_network):
    """Test saving and loading a single network."""
    path = temp_dir / "network.pt"
    
    # Save network
    save_network(simple_network, path)
    assert path.exists()
    
    # Create new network with same architecture
    new_network = torch.nn.Sequential(
        torch.nn.Linear(10, 5),
        torch.nn.ReLU(),
        torch.nn.Linear(5, 1)
    )
    
    # Load network
    load_network(new_network, path)
    
    # Check parameters match
    for p1, p2 in zip(simple_network.parameters(), new_network.parameters()):
        assert torch.allclose(p1, p2)


def test_save_load_optimizer(temp_dir, simple_network):
    """Test saving and loading a single optimizer."""
    # Create optimizer
    optimizer = torch.optim.Adam(simple_network.parameters(), lr=0.001)
    
    path = temp_dir / "optimizer.pt"
    
    # Do a few optimization steps to create state
    x = torch.randn(5, 10)
    y = torch.randn(5, 1)
    optimizer.zero_grad()
    loss = torch.nn.MSELoss()(simple_network(x), y)
    loss.backward()
    optimizer.step()
    
    # Save optimizer
    save_optimizer(optimizer, path)
    assert path.exists()
    
    # Create new optimizer with same parameters
    new_network = torch.nn.Sequential(
        torch.nn.Linear(10, 5),
        torch.nn.ReLU(),
        torch.nn.Linear(5, 1)
    )
    new_optimizer = torch.optim.Adam(new_network.parameters(), lr=0.001)
    
    # Load optimizer
    load_optimizer(new_optimizer, path)
    
    # Check state dicts match
    assert set(optimizer.state_dict().keys()) == set(new_optimizer.state_dict().keys())


def test_save_load_checkpoint_single(temp_dir, simple_network):
    """Test saving and loading a checkpoint with single network and optimizer."""
    # Create optimizer
    optimizer = torch.optim.Adam(simple_network.parameters(), lr=0.001)
    
    path = temp_dir / "checkpoint.pt"
    
    # Save checkpoint
    save_checkpoint(
        path,
        network=simple_network,
        optimizer=optimizer,
        iteration=100,
        metadata={"loss": 0.5, "accuracy": 0.95}
    )
    assert path.exists()
    
    # Create new network and optimizer
    new_network = torch.nn.Sequential(
        torch.nn.Linear(10, 5),
        torch.nn.ReLU(),
        torch.nn.Linear(5, 1)
    )
    new_optimizer = torch.optim.Adam(new_network.parameters(), lr=0.001)
    
    # Load checkpoint
    checkpoint = load_checkpoint(path, network=new_network, optimizer=new_optimizer)
    
    # Check metadata
    assert checkpoint["iteration"] == 100
    assert checkpoint["metadata"]["loss"] == 0.5
    assert checkpoint["metadata"]["accuracy"] == 0.95
    
    # Check parameters match
    for p1, p2 in zip(simple_network.parameters(), new_network.parameters()):
        assert torch.allclose(p1, p2)


def test_save_load_checkpoint_multiple(temp_dir):
    """Test saving and loading a checkpoint with multiple networks and optimizers."""
    path = temp_dir / "multi_checkpoint.pt"
    
    # Create multiple networks and optimizers
    encoder = torch.nn.Linear(10, 5)
    decoder = torch.nn.Linear(5, 10)
    opt_enc = torch.optim.Adam(encoder.parameters(), lr=0.001)
    opt_dec = torch.optim.SGD(decoder.parameters(), lr=0.01)
    
    # Save checkpoint
    save_checkpoint(
        path,
        network={"encoder": encoder, "decoder": decoder},
        optimizer={"opt_enc": opt_enc, "opt_dec": opt_dec},
        iteration=50,
        metadata={"test": "value"}
    )
    assert path.exists()
    
    # Create new networks and optimizers
    new_encoder = torch.nn.Linear(10, 5)
    new_decoder = torch.nn.Linear(5, 10)
    new_opt_enc = torch.optim.Adam(new_encoder.parameters(), lr=0.001)
    new_opt_dec = torch.optim.SGD(new_decoder.parameters(), lr=0.01)
    
    # Load checkpoint
    checkpoint = load_checkpoint(
        path,
        network={"encoder": new_encoder, "decoder": new_decoder},
        optimizer={"opt_enc": new_opt_enc, "opt_dec": new_opt_dec}
    )
    
    # Check metadata
    assert checkpoint["iteration"] == 50
    assert checkpoint["metadata"]["test"] == "value"
    
    # Check parameters match
    for p1, p2 in zip(encoder.parameters(), new_encoder.parameters()):
        assert torch.allclose(p1, p2)
    for p1, p2 in zip(decoder.parameters(), new_decoder.parameters()):
        assert torch.allclose(p1, p2)


def test_save_load_checkpoint_no_optimizer(temp_dir, simple_network):
    """Test saving and loading a checkpoint without optimizer."""
    path = temp_dir / "checkpoint_no_opt.pt"
    
    # Save checkpoint without optimizer
    save_checkpoint(
        path,
        network=simple_network,
        iteration=100,
        metadata={"loss": 0.5}
    )
    assert path.exists()
    
    # Create new network
    new_network = torch.nn.Sequential(
        torch.nn.Linear(10, 5),
        torch.nn.ReLU(),
        torch.nn.Linear(5, 1)
    )
    
    # Load checkpoint without optimizer
    checkpoint = load_checkpoint(path, network=new_network)
    
    # Check metadata
    assert checkpoint["iteration"] == 100
    assert checkpoint["metadata"]["loss"] == 0.5
    assert "optimizers" not in checkpoint
    
    # Check parameters match
    for p1, p2 in zip(simple_network.parameters(), new_network.parameters()):
        assert torch.allclose(p1, p2)


def test_load_checkpoint_missing_network(temp_dir, simple_network):
    """Test loading checkpoint with missing network key."""
    path = temp_dir / "checkpoint.pt"
    
    # Save checkpoint with default name
    save_checkpoint(path, network=simple_network)
    
    # Try to load with different network name
    new_network = torch.nn.Linear(10, 5)
    
    with pytest.raises(KeyError, match="Network 'different_name' not found in checkpoint"):
        load_checkpoint(path, network={"different_name": new_network})


def test_load_checkpoint_sequence_format(temp_dir, simple_network):
    """Test loading checkpoint with sequence format for networks."""
    path = temp_dir / "checkpoint_seq.pt"
    
    # Save with sequence format
    net1 = simple_network
    net2 = torch.nn.Linear(10, 5)
    save_checkpoint(path, network=[net1, net2])
    
    # Load with sequence format
    new_net1 = torch.nn.Sequential(
        torch.nn.Linear(10, 5),
        torch.nn.ReLU(),
        torch.nn.Linear(5, 1)
    )
    new_net2 = torch.nn.Linear(10, 5)
    
    checkpoint = load_checkpoint(path, network=[new_net1, new_net2])
    
    # Check parameters match
    for p1, p2 in zip(net1.parameters(), new_net1.parameters()):
        assert torch.allclose(p1, p2)
    for p1, p2 in zip(net2.parameters(), new_net2.parameters()):
        assert torch.allclose(p1, p2)