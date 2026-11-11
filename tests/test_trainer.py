# -------------------------------------------------------------------------------- #
import csv
import pytest
import torch

from unittest.mock import patch

from sciml.core.checkpoint import Checkpoint
from sciml.core.objective import Objective
from sciml.core.strategy import Strategy
from sciml.core.trainer import Trainer
from sciml.interfaces import CallbackBase
from sciml.implementations.losses import Residual
from sciml.implementations.samplers import FixedPoints
from conftest import LinearModel, SquaredLoss, ConstantSampler

# -------------------------------------------------------------------------------- #
class RecordingCallback(CallbackBase):
    """Records the order in which trainer callbacks are invoked."""

    def __init__(self) -> None:
        self.events = []

    def on_train_start(self) -> None:
        self.events.append("train_start")

    def on_iteration_start(self) -> None:
        self.events.append("iteration_start")

    def on_iteration_end(self) -> None:
        self.events.append("iteration_end")

    def on_train_end(self) -> None:
        self.events.append("train_end")

    def on_exception(self) -> None:
        self.events.append("exception")

class BrokenSampler():
    """Sampler that always fails, to exercise the exception path of Trainer.fit."""

    def next(self):
        raise RuntimeError("boom")

# -------------------------------------------------------------------------------- #
def test_fit_reduces_the_loss():
    torch.manual_seed(0)

    model = LinearModel()
    torch.nn.init.constant_(model.network.weight, 5.0)

    objective = Objective(name="obj", sampler=ConstantSampler(), losses=[SquaredLoss(name="l", weight=1.0)])
    strategy = Strategy(name="s", optimizer=torch.optim.SGD(model.network.parameters(), lr=0.1))

    trainer = Trainer(model=model, objectives=[objective], strategies=[strategy])

    initial_weight = abs(model.network.weight.item())
    trainer.fit(num_iterations=50)

    assert abs(model.network.weight.item()) < initial_weight

def test_callbacks_are_invoked_in_order():
    model = LinearModel()
    objective = Objective(name="obj", sampler=ConstantSampler(), losses=[SquaredLoss(name="l", weight=1.0)])
    strategy = Strategy(name="s", optimizer=torch.optim.SGD(model.network.parameters(), lr=0.1))
    callback = RecordingCallback()

    trainer = Trainer(model=model, objectives=[objective], strategies=[strategy], callbacks=[callback])
    trainer.fit(num_iterations=2)

    assert callback.events == [
        "train_start",
        "iteration_start", "iteration_end",
        "iteration_start", "iteration_end",
        "train_end",
    ]

def test_on_exception_callback_is_invoked_and_error_reraised():
    model = LinearModel()
    objective = Objective(name="obj", sampler=BrokenSampler(), losses=[SquaredLoss(name="l", weight=1.0)])
    strategy = Strategy(name="s", optimizer=torch.optim.SGD(model.network.parameters(), lr=0.1))
    callback = RecordingCallback()

    trainer = Trainer(model=model, objectives=[objective], strategies=[strategy], callbacks=[callback])

    with pytest.raises(RuntimeError):
        trainer.fit(num_iterations=1)

    assert "exception" in callback.events
    assert "train_end" not in callback.events

def test_results_path_writes_a_log_per_strategy(tmp_path):
    model = LinearModel()
    objective = Objective(name="obj", sampler=ConstantSampler(), losses=[SquaredLoss(name="l", weight=1.0)])
    strategy = Strategy(name="s", optimizer=torch.optim.SGD(model.network.parameters(), lr=0.1))

    trainer = Trainer(
        model=model, objectives=[objective], strategies=[strategy], results_path=str(tmp_path),
    )
    trainer.fit(num_iterations=3)

    lines = (tmp_path / "s_log.csv").read_text().splitlines()
    assert len(lines) == 1 + 3   # header + one row per iteration

def test_periodic_and_final_checkpoints_across_fit_calls(tmp_path):
    model = LinearModel()
    objective = Objective("obj", ConstantSampler(), [SquaredLoss(name="loss", weight=1.0)])
    strategies = [
        Strategy(name, torch.optim.Adam(model.network.parameters()))
        for name in ("first", "second")
    ]
    checkpoint = Checkpoint(tmp_path, model.network, strategies)
    trainer = Trainer(model, [objective], strategies, checkpoint=checkpoint, checkpoint_interval=2)

    with patch.object(checkpoint, "save", wraps=checkpoint.save) as save:
        trainer.fit(3)
        trainer.fit(1)
        assert [call.args[0] for call in save.call_args_list] == [2, 3, 4]

    for iteration in (2, 3, 4):
        assert checkpoint.load(iteration)["iteration"] == iteration
        for strategy in strategies:
            state = strategy.optimizer.state[model.network.weight]
            assert state["step"].item() == iteration

def test_invalid_checkpoint_interval_raises():
    with pytest.raises(ValueError, match="checkpoint_interval"):
        Trainer(LinearModel(), [], [], checkpoint_interval=0)

def test_validation_after_optimization_and_at_end_of_fit(tmp_path):
    model = LinearModel()
    torch.nn.init.ones_(model.network.weight)
    strategy = Strategy("sgd", torch.optim.SGD(model.network.parameters(), lr=0.25))
    loss = SquaredLoss(name="mse", weight=1.0)
    training = Objective("train", FixedPoints({"x": [1]}), [loss])
    validation = [Objective("held_out", FixedPoints({"x": [2]}), [loss])]
    trainer = Trainer(model, [training], [strategy], results_path=str(tmp_path),
                      validation=validation, validation_interval=2)
    trainer.fit(3)
    trainer.fit(1)

    with (tmp_path / "validation_log.csv").open() as file:
        rows = list(csv.DictReader(file))
    assert [int(row["iteration"]) for row in rows] == [2, 3, 4]
    for row in rows:
        weight = 0.5 ** int(row["iteration"])
        assert float(row["held_out.mse"]) == pytest.approx((2 * weight) ** 2)
    assert "held_out" not in (tmp_path / "sgd_log.csv").read_text()

def test_validation_supports_input_derivatives_and_restores_modes(tmp_path):
    class CubicModel(LinearModel):
        def compute(self, context):
            assert not self.network.training
            context.requires_grad("x")
            context["u"] = self.network(context["x"].pow(3))

    model = CubicModel()
    model.network = torch.nn.Sequential(torch.nn.Linear(1, 1, bias=False), torch.nn.Dropout())
    torch.nn.init.ones_(model.network[0].weight)
    model.network[0].eval()
    residual = Residual("mse", 1.0, lambda ctx: ctx.partial("u", "x", order=2)[0])
    physics = Objective("physics", FixedPoints({"x": [1, 2]}), [residual])
    Trainer(model, [], [], validation=[physics], results_path=str(tmp_path)).fit(1)

    with (tmp_path / "validation_log.csv").open() as file:
        row, = csv.DictReader(file)
    assert float(row["physics.mse"]) == 90.0
    assert model.network.training and model.network[1].training
    assert not model.network[0].training
    assert model.network[0].weight.grad is None

# -------------------------------------------------------------------------------- #
