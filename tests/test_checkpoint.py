# -------------------------------------------------------------------------------- #
import pytest
import torch

from sciml.utils.checkpoint import save_checkpoint, load_checkpoint

# -------------------------------------------------------------------------------- #
def test_save_and_load_round_trip(tmp_path):
    path = str(tmp_path / "checkpoint.pt")

    net = torch.nn.Linear(2, 1)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

    # Take a few steps so the optimizer state (e.g. Adam moments) is non-trivial
    for _ in range(3):
        optimizer.zero_grad()
        loss = (net(torch.randn(4, 2)) ** 2).mean()
        loss.backward()
        optimizer.step()

    save_checkpoint(path, net, {"adam": optimizer}, metadata={"iteration": 3, "note": "test"})

    net2 = torch.nn.Linear(2, 1)
    optimizer2 = torch.optim.Adam(net2.parameters(), lr=1e-3)
    metadata = load_checkpoint(path, net2, {"adam": optimizer2})

    assert torch.equal(net.weight, net2.weight)
    assert optimizer2.state_dict()["state"].keys() == optimizer.state_dict()["state"].keys()
    assert metadata == {"iteration": 3, "note": "test"}

def test_load_raises_on_optimizer_name_mismatch(tmp_path):
    path = str(tmp_path / "checkpoint.pt")

    net = torch.nn.Linear(2, 1)
    save_checkpoint(path, net, {"adam": torch.optim.Adam(net.parameters())})

    net2 = torch.nn.Linear(2, 1)
    with pytest.raises(ValueError):
        load_checkpoint(path, net2, {"sgd": torch.optim.SGD(net2.parameters(), lr=0.1)})

# -------------------------------------------------------------------------------- #
