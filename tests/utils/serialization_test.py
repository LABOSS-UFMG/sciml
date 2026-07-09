import torch

from sciml.utils.serialization import (
    load_checkpoint,
    load_network,
    save_checkpoint,
    save_network,
)


def test_save_and_load_network_restores_parameters(tmp_path):
    network = torch.nn.Linear(1, 1)
    with torch.no_grad():
        network.weight.fill_(2.0)
        network.bias.fill_(3.0)

    path = tmp_path / "nested" / "network.pt"
    save_network(network, path)

    with torch.no_grad():
        network.weight.zero_()
        network.bias.zero_()

    load_network(network, path)

    assert torch.equal(network.weight, torch.tensor([[2.0]]))
    assert torch.equal(network.bias, torch.tensor([3.0]))


def test_save_and_load_checkpoint_restores_network_and_metadata(tmp_path):
    network = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(network.parameters(), lr=0.1)
    path = tmp_path / "checkpoint.pt"

    with torch.no_grad():
        network.weight.fill_(4.0)
        network.bias.fill_(5.0)
    save_checkpoint(
        path,
        network=network,
        optimizer=optimizer,
        epoch=7,
        metadata={"case": "test"},
    )

    with torch.no_grad():
        network.weight.zero_()
        network.bias.zero_()
    loaded = load_checkpoint(path, network=network, optimizer=optimizer)

    assert torch.equal(network.weight, torch.tensor([[4.0]]))
    assert torch.equal(network.bias, torch.tensor([5.0]))
    assert loaded["epoch"] == 7
    assert loaded["metadata"] == {"case": "test"}
    assert "optimizer" in loaded
