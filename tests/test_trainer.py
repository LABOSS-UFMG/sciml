# -------------------------------------------------------------------------------- #
import pytest
import torch

from sciml.core.objective import Objective
from sciml.core.strategy import Strategy
from sciml.core.trainer import Trainer
from sciml.interfaces import CallbackBase
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

# -------------------------------------------------------------------------------- #
