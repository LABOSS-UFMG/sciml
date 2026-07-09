import numpy as np
import pytest
import torch

from sciml.core.contracts import Batch
from sciml.samplers.latin_hypercube import LatinHypercube


def test_latin_hypercube_returns_batched_inputs_targets_and_cycles():
    sampler = LatinHypercube(
        name="data",
        n_points=5,
        batch_size=2,
        bounds=[(0.0, 1.0), (10.0, 20.0)],
        margin=[0.1, 0.1],
        insertions=[(1, -1.0)],
        input_key="x",
        target_key="y",
        target_fn=lambda x: np.sum(x, axis=1, keepdims=True),
        seed=42,
        dtype=torch.float64,
    )

    first = sampler.next()
    second = sampler.next()
    third = sampler.next()
    wrapped = sampler.next()

    assert isinstance(first, Batch)
    assert first.inputs["x"].shape == (2, 3)
    assert second.inputs["x"].shape == (2, 3)
    assert third.inputs["x"].shape == (1, 3)
    assert torch.equal(first.inputs["x"], wrapped.inputs["x"])
    assert torch.all(first.inputs["x"][:, 0] >= 0.1)
    assert torch.all(first.inputs["x"][:, 0] <= 0.9)
    assert torch.all(first.inputs["x"][:, 1] == -1.0)
    assert torch.all(first.inputs["x"][:, 2] >= 11.0)
    assert torch.all(first.inputs["x"][:, 2] <= 19.0)
    assert torch.equal(first.targets["y"], first.inputs["x"].sum(dim=1, keepdim=True))


def test_latin_hypercube_rejects_target_function_without_target_key():
    with pytest.raises(ValueError):
        LatinHypercube(
            name="data",
            n_points=5,
            batch_size=2,
            bounds=[(0.0, 1.0)],
            target_fn=lambda x: x,
        )


def test_latin_hypercube_rejects_duplicate_insertions():
    with pytest.raises(ValueError):
        LatinHypercube(
            name="data",
            n_points=5,
            batch_size=2,
            bounds=[(0.0, 1.0)],
            insertions=[(0, 0.0), (0, 1.0)],
        )
