# -------------------------------------------------------------------------------- #
import pytest
import torch

from sciml.implementations.samplers import LatinHypercube

# -------------------------------------------------------------------------------- #
def test_default_keys_and_shapes():
    sampler = LatinHypercube(dim=2, num_points=20, batch_size=5, seed=0)
    context = sampler.next()

    assert context["x1"].shape == (5, 1)
    assert context["x2"].shape == (5, 1)

def test_bounds_are_respected():
    sampler = LatinHypercube(
        dim=1, num_points=50, batch_size=50,
        bounds=[(0, (2.0, 4.0))], seed=0,
    )
    x = sampler.next()["x1"]

    assert torch.all(x >= 2.0) and torch.all(x <= 4.0)

def test_insertion_fixes_dimension():
    sampler = LatinHypercube(
        dim=2, num_points=10, batch_size=10,
        input_keys=["x", "t"], insertions=[(0, 0.0)], seed=0,
    )
    context = sampler.next()

    assert torch.all(context["x"] == 0.0)

def test_target_fn_populates_targets():
    sampler = LatinHypercube(
        dim=1, num_points=10, batch_size=10,
        input_keys=["x"], target_keys=["y"],
        target_fn=lambda x: x * 2, seed=0,
    )
    context = sampler.next()

    assert torch.allclose(context["y"], context["x"] * 2)

def test_positional_targets_bounds_and_insertions():
    input_keys = ["x", "t"]
    target_keys = ["y"]
    bounds = [(0, (2.0, 4.0))]
    insertions = [(1, 5.0)]

    def target_fn(x, t):
        return 2 * x + t

    sampler = LatinHypercube(
        2, 10, 10, input_keys, target_keys, target_fn, bounds, insertions, seed=0,
    )
    context = sampler.next()

    assert torch.all((context["x"] >= 2.0) & (context["x"] <= 4.0))
    assert torch.all(context["t"] == 5.0)
    torch.testing.assert_close(context["y"], 2 * context["x"] + context["t"])

def test_cycles_back_to_first_batch():
    sampler = LatinHypercube(dim=1, num_points=10, batch_size=5, seed=0)

    first = sampler.next()["x1"]
    sampler.next()
    third = sampler.next()["x1"]   # wraps back around after 2 batches

    assert torch.equal(first, third)

@pytest.mark.parametrize(
    "kwargs",
    [
        dict(dim=0, num_points=10, batch_size=5),
        dict(dim=1, num_points=0, batch_size=5),
        dict(dim=1, num_points=10, batch_size=0),
        dict(dim=1, num_points=10, batch_size=20),
    ],
)
def test_invalid_arguments_raise(kwargs):
    with pytest.raises(ValueError):
        LatinHypercube(**kwargs)

# -------------------------------------------------------------------------------- #
