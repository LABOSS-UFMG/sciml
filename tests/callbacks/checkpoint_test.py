from pathlib import Path

import torch

from sciml.callbacks.checkpoint import Checkpoint


def _network_and_optimizer():
    network = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(network.parameters(), lr=0.1)
    return network, optimizer


def test_checkpoint_saves_at_frequency_and_at_train_end(tmp_path):
    network, optimizer = _network_and_optimizer()
    checkpoint = Checkpoint(directory=str(tmp_path), frequency=2)

    checkpoint.on_iteration_end(network, optimizer, iteration=1)
    checkpoint.on_iteration_end(network, optimizer, iteration=2)
    checkpoint.on_train_end(network, optimizer, iteration=3)

    assert not (tmp_path / "checkpoint_1.pt").exists()
    assert (tmp_path / "checkpoint_2.pt").exists()
    assert (tmp_path / "checkpoint_3.pt").exists()


def test_checkpoint_saves_recovery_on_exception(tmp_path):
    network, optimizer = _network_and_optimizer()
    checkpoint = Checkpoint(directory=str(tmp_path), frequency=10)

    checkpoint.on_exception(network, optimizer, iteration=7)

    assert (tmp_path / "recovery_7.pt").exists()


def test_checkpoint_rejects_invalid_frequency(tmp_path):
    try:
        Checkpoint(directory=str(tmp_path), frequency=0)
    except ValueError as error:
        assert "positive" in str(error)
    else:
        raise AssertionError("Checkpoint accepted a non-positive frequency")
