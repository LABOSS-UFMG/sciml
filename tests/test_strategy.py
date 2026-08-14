# -------------------------------------------------------------------------------- #
import torch

from sciml.core.objective import Objective
from sciml.core.strategy import Strategy
from conftest import LinearModel, SquaredLoss, ConstantSampler

# -------------------------------------------------------------------------------- #
def test_step_updates_parameters():
    model = LinearModel()
    objective = Objective(name="obj", sampler=ConstantSampler(), losses=[SquaredLoss(name="l", weight=1.0)])
    strategy = Strategy(name="s", optimizer=torch.optim.SGD(model.network.parameters(), lr=0.1))

    before = model.network.weight.item()
    step = strategy.step(1, model, [objective])
    after = model.network.weight.item()

    assert step is not None
    assert step.name == "s"
    assert after != before

def test_step_returns_none_when_disabled():
    model = LinearModel()
    objective = Objective(name="obj", sampler=ConstantSampler(), losses=[SquaredLoss(name="l", weight=1.0)])
    strategy = Strategy(
        name="s",
        optimizer=torch.optim.SGD(model.network.parameters(), lr=0.1),
        enable=lambda iteration: False,
    )

    before = model.network.weight.item()
    step = strategy.step(1, model, [objective])
    after = model.network.weight.item()

    assert step is None
    assert after == before

def test_step_filters_by_objective_names():
    model = LinearModel()
    included = Objective(name="included", sampler=ConstantSampler(), losses=[SquaredLoss(name="l1", weight=1.0)])
    excluded = Objective(name="excluded", sampler=ConstantSampler(), losses=[SquaredLoss(name="l2", weight=1.0)])

    strategy = Strategy(
        name="s",
        optimizer=torch.optim.SGD(model.network.parameters(), lr=0.1),
        objective_names=["included"],
    )
    step = strategy.step(1, model, [included, excluded])

    assert [evaluation.name for evaluation in step.evaluations] == ["included"]

# -------------------------------------------------------------------------------- #
