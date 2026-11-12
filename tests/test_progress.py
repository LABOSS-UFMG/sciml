# -------------------------------------------------------------------------------- #
import csv

import pytest
import torch

from sciml.core.objective import Objective
from sciml.core.progress import Progress
from sciml.core.strategy import Strategy
from conftest import LinearModel, SquaredLoss, ConstantSampler

# -------------------------------------------------------------------------------- #
def _make_strategies(model, names):
    return [
        Strategy(name=name, optimizer=torch.optim.SGD(model.network.parameters(), lr=0.1))
        for name in names
    ]

def test_update_tracks_active_and_inactive_strategies():
    model = LinearModel()
    objective = Objective(name="obj", sampler=ConstantSampler(), losses=[SquaredLoss(name="l", weight=1.0)])
    strategy_a, strategy_b = _make_strategies(model, ["a", "b"])

    progress = Progress([strategy_a, strategy_b])
    step_a = strategy_a.step(1, model, [objective])   # only "a" runs this iteration

    progress.update([step_a], 1)

    assert progress.active == {"a": True, "b": False}
    assert "a" in progress.current
    assert "b" not in progress.current

def test_render_live_tags_inactive_strategies():
    model = LinearModel()
    objective = Objective(name="obj", sampler=ConstantSampler(), losses=[SquaredLoss(name="l", weight=1.0)])
    strategy_a, strategy_b = _make_strategies(model, ["a", "b"])

    progress = Progress([strategy_a, strategy_b])
    step_a = strategy_a.step(1, model, [objective])
    progress.update([step_a], 1)

    block = progress.render_live(1)

    assert "[active]" in block
    assert "[inactive]" in block

def test_log_writes_header_once_and_skips_inactive_strategy(tmp_path):
    model = LinearModel()
    objective = Objective(name="obj", sampler=ConstantSampler(), losses=[SquaredLoss(name="l", weight=1.0)])
    strategy_a, strategy_b = _make_strategies(model, ["a", "b"])

    progress = Progress([strategy_a, strategy_b], path=tmp_path, filename="history.csv")

    for iteration in range(1, 4):
        steps = [strategy_a.step(iteration, model, [objective])]   # "b" never runs
        progress.update(steps, iteration)
        assert not (tmp_path / "a_history.csv").exists()

    progress.save()
    progress.save()

    lines = (tmp_path / "a_history.csv").read_text().splitlines()
    assert lines[0].split(",")[:3] == ["iteration", "ratio", "objective"]
    assert len(lines) == 1 + 3   # header + one row per iteration

    assert not (tmp_path / "b_history.csv").exists()

def test_log_preserves_same_named_losses_from_different_objectives(tmp_path):
    model = LinearModel()
    model.network.weight.data.fill_(1.0)
    strategy, = _make_strategies(model, ["adam"])
    objectives = [
        Objective(name, ConstantSampler(value=x), [SquaredLoss(name="mse", weight=1.0)])
        for name, x in [("interior", 2.0), ("contorno", 3.0)]
    ]
    step = strategy.step(1, model, objectives)
    progress = Progress([strategy], path=tmp_path)
    progress.update([step], 1)
    progress.save()

    path = tmp_path / "adam_log.csv"
    with path.open(newline="") as file:
        row, = csv.DictReader(file)
    assert float(row["interior.mse"]) == 4.0
    assert float(row["contorno.mse"]) == 9.0
    assert float(row["objective"]) == 13.0

    legacy = "iteration,ratio,objective,mse\n1,1,13,9\n"
    path.write_text(legacy)
    progress.update([step], 2)
    with pytest.raises(ValueError, match="Incompatible history columns"):
        progress.save()
    assert path.read_text() == legacy

# -------------------------------------------------------------------------------- #
