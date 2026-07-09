import pytest
import torch

from sciml.core.contracts import Batch
from sciml.core.loss import LossBase
from sciml.core.sampler import SamplerBase
from sciml.trainers.standard import Standard


class FixedSampler(SamplerBase):
    def __init__(self, name, batch):
        self.name = name
        self.batch = batch

    def next(self):
        return self.batch


class MeanOutputLoss(LossBase):
    def __init__(self, name="loss"):
        self.name = name
        self.weight = 1.0

    def evaluate(self, network, batch):
        return torch.mean(network(batch.inputs["x"]) ** 2)


def test_standard_rejects_mismatched_loss_and_sampler_names():
    network = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(network.parameters(), lr=0.1)
    loss = MeanOutputLoss(name="loss")
    sampler = FixedSampler(name="other", batch=Batch(inputs={"x": torch.ones(2, 1)}))

    with pytest.raises(ValueError):
        Standard(network=network, losses=[loss], samplers=[sampler], optimizer=optimizer)


def test_standard_fit_runs_requested_iterations():
    network = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(network.parameters(), lr=0.1)
    loss = MeanOutputLoss(name="loss")
    sampler = FixedSampler(name="loss", batch=Batch(inputs={"x": torch.ones(2, 1)}))
    trainer = Standard(network=network, losses=[loss], samplers=[sampler], optimizer=optimizer)

    trainer.fit(iterations=2)

    assert trainer._iteration == 2
