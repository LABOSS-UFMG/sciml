import pytest
import torch
import tempfile
import os
from pathlib import Path
import warnings

from sciml.callbacks.checkpoint import Checkpoint
from sciml.utils import serialization


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
        yield tmpdir


def test_checkpoint_init_valid(temp_dir, simple_network, simple_optimizer):
    """Test Checkpoint initialization with valid parameters."""
    checkpoint = Checkpoint(
        directory=temp_dir,
        frequency=100,
        network=simple_network,
        optimizer=simple_optimizer,
        filename_template="checkpoint_{iteration}.pt"
    )
    
    assert checkpoint.directory == temp_dir.rstrip("/")
    assert checkpoint.frequency == 100
    assert checkpoint.network is simple_network
    assert checkpoint.optimizer is simple_optimizer
    assert checkpoint.filename_template == "checkpoint_{iteration}.pt"


def test_checkpoint_init_without_optimizer(temp_dir, simple_network):
    """Test Checkpoint initialization without optimizer."""
    checkpoint = Checkpoint(
        directory=temp_dir,
        frequency=50,
        network=simple_network,
        optimizer=None
    )
    
    assert checkpoint.optimizer is None


def test_checkpoint_init_invalid_frequency(temp_dir, simple_network):
    """Test Checkpoint initialization with invalid frequency."""
    with pytest.raises(ValueError, match="frequency must be a positive integer"):
        Checkpoint(
            directory=temp_dir,
            frequency=0,
            network=simple_network
        )
    
    with pytest.raises(ValueError, match="frequency must be a positive integer"):
        Checkpoint(
            directory=temp_dir,
            frequency=-5,
            network=simple_network
        )


def test_checkpoint_init_invalid_template(temp_dir, simple_network):
    """Test Checkpoint initialization with invalid filename template."""
    with pytest.raises(ValueError, match="filename_template must contain the '{iteration}' placeholder"):
        Checkpoint(
            directory=temp_dir,
            frequency=100,
            network=simple_network,
            filename_template="checkpoint.pt"
        )


def test_checkpoint_path_for(temp_dir, simple_network):
    """Test _path_for method generates correct path."""
    checkpoint = Checkpoint(
        directory=temp_dir,
        frequency=100,
        network=simple_network,
        filename_template="checkpoint_{iteration}.pt"
    )
    
    path = checkpoint._path_for(42)
    expected = f"{temp_dir.rstrip('/')}/checkpoint_42.pt"
    assert path == expected


def test_checkpoint_path_for_custom_template(temp_dir, simple_network):
    """Test _path_for with custom filename template."""
    checkpoint = Checkpoint(
        directory=temp_dir,
        frequency=100,
        network=simple_network,
        filename_template="model_iter_{iteration}.pt"
    )
    
    path = checkpoint._path_for(42)
    expected = f"{temp_dir.rstrip('/')}/model_iter_42.pt"
    assert path == expected


def test_checkpoint_on_iteration_end(temp_dir, simple_network, simple_optimizer):
    """Test checkpoint saving at iteration end when frequency matches."""
    checkpoint = Checkpoint(
        directory=temp_dir,
        frequency=10,
        network=simple_network,
        optimizer=simple_optimizer
    )
    
    # Should NOT save at iteration 5 (not multiple of 10)
    checkpoint.on_iteration_end(iteration=5)
    assert not os.path.exists(checkpoint._path_for(5))
    
    # Should save at iteration 10 (multiple of 10)
    checkpoint.on_iteration_end(iteration=10)
    assert os.path.exists(checkpoint._path_for(10))
    
    # Should save at iteration 20 (multiple of 10)
    checkpoint.on_iteration_end(iteration=20)
    assert os.path.exists(checkpoint._path_for(20))
    
    # Verify checkpoint can be loaded
    new_network = torch.nn.Sequential(
        torch.nn.Linear(10, 5),
        torch.nn.ReLU(),
        torch.nn.Linear(5, 1)
    )
    new_optimizer = torch.optim.Adam(new_network.parameters(), lr=0.001)
    
    checkpoint_data = serialization.load_checkpoint(
        checkpoint._path_for(10),
        network=new_network,
        optimizer=new_optimizer
    )
    
    # Check that parameters match
    for p1, p2 in zip(simple_network.parameters(), new_network.parameters()):
        assert torch.allclose(p1, p2)


def test_checkpoint_on_train_end(temp_dir, simple_network, simple_optimizer):
    """Test checkpoint saving at train end."""
    checkpoint = Checkpoint(
        directory=temp_dir,
        frequency=10,
        network=simple_network,
        optimizer=simple_optimizer
    )
    
    # If iteration is multiple of frequency, on_train_end should NOT save
    # (it would have been saved at the end of the iteration)
    checkpoint.on_iteration_end(iteration=10)
    checkpoint.on_train_end(iteration=10)
    # Only one file should exist (from on_iteration_end)
    files = [f for f in os.listdir(temp_dir) if f.startswith("checkpoint_")]
    assert len(files) == 1
    assert "checkpoint_10.pt" in files
    
    # If iteration is NOT multiple of frequency, on_train_end should save
    checkpoint.on_train_end(iteration=15)
    assert os.path.exists(checkpoint._path_for(15))
    
    # Verify checkpoint can be loaded
    new_network = torch.nn.Sequential(
        torch.nn.Linear(10, 5),
        torch.nn.ReLU(),
        torch.nn.Linear(5, 1)
    )
    new_optimizer = torch.optim.Adam(new_network.parameters(), lr=0.001)
    
    checkpoint_data = serialization.load_checkpoint(
        checkpoint._path_for(15),
        network=new_network,
        optimizer=new_optimizer
    )
    
    # Check that parameters match
    for p1, p2 in zip(simple_network.parameters(), new_network.parameters()):
        assert torch.allclose(p1, p2)


def test_checkpoint_on_exception(temp_dir, simple_network, simple_optimizer):
    """Test checkpoint saving on exception."""
    checkpoint = Checkpoint(
        directory=temp_dir,
        frequency=10,
        network=simple_network,
        optimizer=simple_optimizer
    )
    
    # Trigger exception recovery
    checkpoint.on_exception(iteration=42)
    
    # Check recovery file was created
    recovery_path = f"{temp_dir.rstrip('/')}/recovery_42.pt"
    assert os.path.exists(recovery_path)
    
    # Verify recovery checkpoint can be loaded
    new_network = torch.nn.Sequential(
        torch.nn.Linear(10, 5),
        torch.nn.ReLU(),
        torch.nn.Linear(5, 1)
    )
    new_optimizer = torch.optim.Adam(new_network.parameters(), lr=0.001)
    
    checkpoint_data = serialization.load_checkpoint(
        recovery_path,
        network=new_network,
        optimizer=new_optimizer
    )
    
    # Check that parameters match
    for p1, p2 in zip(simple_network.parameters(), new_network.parameters()):
        assert torch.allclose(p1, p2)


def test_checkpoint_no_op_methods(temp_dir, simple_network):
    """Test that on_train_start and on_iteration_start are no-ops."""
    checkpoint = Checkpoint(
        directory=temp_dir,
        frequency=10,
        network=simple_network
    )
    
    # These should not raise any errors
    checkpoint.on_train_start()
    checkpoint.on_iteration_start()
    
    # No files should be created
    assert len(os.listdir(temp_dir)) == 0


def test_checkpoint_with_multiple_networks(temp_dir):
    """Test Checkpoint with multiple networks."""
    encoder = torch.nn.Linear(10, 5)
    decoder = torch.nn.Linear(5, 10)
    opt_enc = torch.optim.Adam(encoder.parameters(), lr=0.001)
    opt_dec = torch.optim.SGD(decoder.parameters(), lr=0.01)
    
    checkpoint = Checkpoint(
        directory=temp_dir,
        frequency=5,
        network={"encoder": encoder, "decoder": decoder},
        optimizer={"opt_enc": opt_enc, "opt_dec": opt_dec}
    )
    
    # Save checkpoint
    checkpoint.on_iteration_end(iteration=5)
    
    # Verify file exists
    assert os.path.exists(checkpoint._path_for(5))
    
    # Load and verify
    new_encoder = torch.nn.Linear(10, 5)
    new_decoder = torch.nn.Linear(5, 10)
    new_opt_enc = torch.optim.Adam(new_encoder.parameters(), lr=0.001)
    new_opt_dec = torch.optim.SGD(new_decoder.parameters(), lr=0.01)
    
    checkpoint_data = serialization.load_checkpoint(
        checkpoint._path_for(5),
        network={"encoder": new_encoder, "decoder": new_decoder},
        optimizer={"opt_enc": new_opt_enc, "opt_dec": new_opt_dec}
    )
    
    # Check parameters match
    for p1, p2 in zip(encoder.parameters(), new_encoder.parameters()):
        assert torch.allclose(p1, p2)
    for p1, p2 in zip(decoder.parameters(), new_decoder.parameters()):
        assert torch.allclose(p1, p2)


def test_checkpoint_with_sequence_networks(temp_dir):
    """Test Checkpoint with sequence of networks."""
    net1 = torch.nn.Linear(10, 5)
    net2 = torch.nn.Linear(5, 3)
    opt1 = torch.optim.Adam(net1.parameters(), lr=0.001)
    opt2 = torch.optim.SGD(net2.parameters(), lr=0.01)
    
    checkpoint = Checkpoint(
        directory=temp_dir,
        frequency=5,
        network=[net1, net2],
        optimizer=[opt1, opt2]
    )
    
    # Save checkpoint
    checkpoint.on_iteration_end(iteration=5)
    
    # Verify file exists
    assert os.path.exists(checkpoint._path_for(5))
    
    # Load and verify
    new_net1 = torch.nn.Linear(10, 5)
    new_net2 = torch.nn.Linear(5, 3)
    new_opt1 = torch.optim.Adam(new_net1.parameters(), lr=0.001)
    new_opt2 = torch.optim.SGD(new_net2.parameters(), lr=0.01)
    
    checkpoint_data = serialization.load_checkpoint(
        checkpoint._path_for(5),
        network=[new_net1, new_net2],
        optimizer=[new_opt1, new_opt2]
    )
    
    # Check parameters match
    for p1, p2 in zip(net1.parameters(), new_net1.parameters()):
        assert torch.allclose(p1, p2)
    for p1, p2 in zip(net2.parameters(), new_net2.parameters()):
        assert torch.allclose(p1, p2)