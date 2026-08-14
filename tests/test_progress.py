# -------------------------------------------------------------------------------- #
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

    progress.update([step_a])

    assert progress.active == {"a": True, "b": False}
    assert "a" in progress.current
    assert "b" not in progress.current

def test_render_live_tags_inactive_strategies():
    model = LinearModel()
    objective = Objective(name="obj", sampler=ConstantSampler(), losses=[SquaredLoss(name="l", weight=1.0)])
    strategy_a, strategy_b = _make_strategies(model, ["a", "b"])

    progress = Progress([strategy_a, strategy_b])
    step_a = strategy_a.step(1, model, [objective])
    progress.update([step_a])

    block = progress.render_live(1)

    assert "[active]" in block
    assert "[inactive]" in block

def test_log_writes_header_once_and_skips_inactive_strategy(tmp_path):
    model = LinearModel()
    objective = Objective(name="obj", sampler=ConstantSampler(), losses=[SquaredLoss(name="l", weight=1.0)])
    strategy_a, strategy_b = _make_strategies(model, ["a", "b"])

    progress = Progress([strategy_a, strategy_b])

    for iteration in range(1, 4):
        steps = [strategy_a.step(iteration, model, [objective])]   # "b" never runs
        progress.update(steps)
        progress.log(str(tmp_path), iteration, steps)

    lines = (tmp_path / "a_log.csv").read_text().splitlines()
    assert lines[0].split(",")[:3] == ["iteration", "ratio", "objective"]
    assert len(lines) == 1 + 3   # header + one row per iteration

    assert not (tmp_path / "b_log.csv").exists()

# -------------------------------------------------------------------------------- #
