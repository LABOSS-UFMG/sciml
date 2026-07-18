import pytest
import torch

from sciml.trainers.trainer import Trainer
from sciml.core.model import ModelBase
from sciml.core.objective import Objective
from sciml.core.strategy import StrategyBase
from sciml.core.validation import Validation
from sciml.core.callback import CallbackBase
from sciml.core.evaluation import Evaluation
from sciml.callbacks.logger import Logger
from sciml.callbacks.checkpoint import Checkpoint


class _FakeModel(ModelBase):
    def compute(self, context):
        return


class _FakeObjective(Objective):
    """Trainer never touches an Objective directly — it only forwards
    the list to each Strategy — so a bare subclass is enough here."""
    pass


class _StubStrategy(StrategyBase):
    """Returns a fixed Evaluation (or raises) instead of performing any
    real optimization, and records every call it receives."""

    def __init__(self, result, name: str = "strategy"):
        super().__init__(name=name)
        self._result = result
        self.calls = []

    def step(self, model, objectives):
        self.calls.append((model, objectives))
        if isinstance(self._result, Exception):
            raise self._result
        return self._result


class _StubValidation(Validation):
    """Bypasses Validation's own (broken) constructor entirely and
    implements `evaluate` with the correct signature declared by the
    abstract contract (`evaluate(self, iteration, model)`)."""

    def __init__(self, result):
        self._result = result
        self.calls = []

    def evaluate(self, iteration, model):
        self.calls.append((iteration, model))
        return self._result


class _SpyLogger(Logger):
    """Bypasses the real Logger's __init__ (which creates a directory on
    disk) and just records every call."""

    def __init__(self):
        self.calls = []

    def on_train_start(self, *args, **kwargs):
        self.calls.append(("train_start", args, kwargs))

    def on_iteration_start(self, *args, **kwargs):
        self.calls.append(("iteration_start", args, kwargs))

    def on_iteration_end(self, iteration, losses, metrics=None):
        self.calls.append(("iteration_end", iteration, losses, metrics))

    def on_train_end(self, *args, **kwargs):
        self.calls.append(("train_end", args, kwargs))

    def on_exception(self, *args, **kwargs):
        self.calls.append(("exception", args, kwargs))


class _SpyCheckpoint(Checkpoint):
    """Bypasses the real Checkpoint's __init__ (which requires a network/
    optimizer and validates a lot) and just records every call."""

    def __init__(self):
        self.calls = []

    def on_train_start(self, *args, **kwargs):
        self.calls.append(("train_start", args, kwargs))

    def on_iteration_start(self, *args, **kwargs):
        self.calls.append(("iteration_start", args, kwargs))

    def on_iteration_end(self, iteration):
        self.calls.append(("iteration_end", iteration))

    def on_train_end(self, iteration):
        self.calls.append(("train_end", iteration))

    def on_exception(self, iteration):
        self.calls.append(("exception", iteration))


class _SpyCallback(CallbackBase):
    def __init__(self):
        self.events = []

    def on_train_start(self, *args, **kwargs):
        self.events.append("train_start")

    def on_iteration_start(self, *args, **kwargs):
        self.events.append("iteration_start")

    def on_iteration_end(self, *args, **kwargs):
        self.events.append("iteration_end")

    def on_train_end(self, *args, **kwargs):
        self.events.append("train_end")

    def on_exception(self, *args, **kwargs):
        self.events.append("exception")


def _evaluation(name="strategy", value=1.0, weight=1.0):
    return Evaluation(
        values={"loss": torch.tensor(value)},
        weights={"loss": weight},
        metadata={"name": name},
    )


def test_rejects_non_model_instance():
    strategy = _StubStrategy(_evaluation())
    with pytest.raises(TypeError):
        Trainer(model="not a model", objectives=[], strategies=[strategy])


def test_rejects_non_iterable_objectives_dtype():
    strategy = _StubStrategy(_evaluation())
    with pytest.raises(TypeError):
        Trainer(model=_FakeModel(), objectives=["not an objective"], strategies=[strategy])


def test_rejects_non_iterable_strategies_dtype():
    with pytest.raises(TypeError):
        Trainer(model=_FakeModel(), objectives=[], strategies=["not a strategy"])


def test_logger_must_be_a_logger_instance_when_given():
    strategy = _StubStrategy(_evaluation())
    with pytest.raises(TypeError):
        Trainer(
            model=_FakeModel(), objectives=[], strategies=[strategy],
            logger="not a logger",
        )


def test_checkpoint_must_be_a_checkpoint_instance_when_given():
    strategy = _StubStrategy(_evaluation())
    with pytest.raises(TypeError):
        Trainer(
            model=_FakeModel(), objectives=[], strategies=[strategy],
            checkpoint="not a checkpoint",
        )


def test_callbacks_must_be_callbackbase_instances_when_given():
    strategy = _StubStrategy(_evaluation())
    with pytest.raises(TypeError):
        Trainer(
            model=_FakeModel(), objectives=[], strategies=[strategy],
            callbacks=["not a callback"],
        )


def test_validations_defaults_to_none():
    strategy = _StubStrategy(_evaluation())
    trainer = Trainer(model=_FakeModel(), objectives=[], strategies=[strategy])

    assert trainer.validations is None


def test_callbacks_defaults_to_none():
    strategy = _StubStrategy(_evaluation())
    trainer = Trainer(model=_FakeModel(), objectives=[], strategies=[strategy])

    assert trainer.callbacks is None


def test_iteration_counter_starts_at_zero():
    strategy = _StubStrategy(_evaluation())
    trainer = Trainer(model=_FakeModel(), objectives=[], strategies=[strategy])

    assert trainer._iteration == 0


def test_validations_type_is_not_checked_at_construction():
    """
    NOTE: unlike `objectives`, `strategies`, `logger`, `checkpoint`, and
    `callbacks`, the constructor never validates `validations` (no
    `checker.is_iterable(validations, dtype=Validation)` call exists).
    This documents that current gap; not urgent, but inconsistent with
    every other parameter.
    """
    strategy = _StubStrategy(_evaluation())
    trainer = Trainer(
        model=_FakeModel(), objectives=[], strategies=[strategy],
        validations=["not a validation"],
    )

    assert trainer.validations == ["not a validation"]


def test_on_train_start_fires_once_for_every_callback_before_the_first_iteration():
    callback = _SpyCallback()
    strategy = _StubStrategy(_evaluation())
    trainer = Trainer(
        model=_FakeModel(), objectives=[], strategies=[strategy],
        callbacks=[callback],
    )

    trainer.fit(num_iterations=1)

    assert callback.events[0] == "train_start"


def test_strategy_step_is_called_with_model_and_objectives():
    objective = _FakeObjective()
    strategy = _StubStrategy(_evaluation())
    trainer = Trainer(
        model=_FakeModel(), objectives=[objective], strategies=[strategy],
    )

    trainer.fit(num_iterations=1)

    assert len(strategy.calls) == 1
    called_model, called_objectives = strategy.calls[0]
    assert called_model is trainer.model
    assert called_objectives == [objective]


def test_iteration_counter_increments_once_per_iteration():
    strategy = _StubStrategy(_evaluation())
    trainer = Trainer(model=_FakeModel(), objectives=[], strategies=[strategy])

    trainer.fit(num_iterations=3)

    assert trainer._iteration == 3


def test_callback_events_fire_in_order_across_iterations():
    callback = _SpyCallback()
    strategy = _StubStrategy(_evaluation())
    trainer = Trainer(
        model=_FakeModel(), objectives=[], strategies=[strategy],
        callbacks=[callback],
    )

    trainer.fit(num_iterations=2)

    assert callback.events == [
        "train_start",
        "iteration_start", "iteration_end",
        "iteration_start", "iteration_end",
        "train_end",
    ]


def test_fit_completes_without_logger_checkpoint_or_validations():
    """The simplest possible usage: no logger, no checkpoint, no
    validations. All three are optional and default to None."""
    strategy = _StubStrategy(_evaluation())
    trainer = Trainer(model=_FakeModel(), objectives=[], strategies=[strategy])

    trainer.fit(num_iterations=1)

    assert trainer._iteration == 1


def test_fit_completes_with_every_optional_component_provided():
    strategy = _StubStrategy(_evaluation())
    validation = _StubValidation(_evaluation(name="val"))
    logger = _SpyLogger()
    checkpoint = _SpyCheckpoint()
    callback = _SpyCallback()

    trainer = Trainer(
        model=_FakeModel(), objectives=[], strategies=[strategy],
        validations=[validation], logger=logger, checkpoint=checkpoint,
        callbacks=[callback],
    )

    trainer.fit(num_iterations=1)

    assert trainer._iteration == 1


def test_validation_evaluate_is_called_with_iteration_and_model():
    strategy = _StubStrategy(_evaluation())
    validation = _StubValidation(_evaluation(name="val"))
    logger = _SpyLogger()
    checkpoint = _SpyCheckpoint()
    trainer = Trainer(
        model=_FakeModel(), objectives=[], strategies=[strategy],
        validations=[validation], logger=logger, checkpoint=checkpoint,
    )

    trainer.fit(num_iterations=1)

    assert validation.calls == [(1, trainer.model)]


def test_metrics_passed_to_logger_are_none_without_validations():
    strategy = _StubStrategy(_evaluation())
    logger = _SpyLogger()
    trainer = Trainer(
        model=_FakeModel(), objectives=[], strategies=[strategy], logger=logger,
    )

    trainer.fit(num_iterations=1)

    iteration_end_call = next(c for c in logger.calls if c[0] == "iteration_end")
    _, iteration, losses, metrics = iteration_end_call
    assert metrics is None


def test_metrics_passed_to_logger_reflect_validation_results():
    strategy = _StubStrategy(_evaluation())
    validation_evaluation = _evaluation(name="val")
    validation = _StubValidation(validation_evaluation)
    logger = _SpyLogger()
    trainer = Trainer(
        model=_FakeModel(), objectives=[], strategies=[strategy],
        validations=[validation], logger=logger,
    )

    trainer.fit(num_iterations=1)

    iteration_end_call = next(c for c in logger.calls if c[0] == "iteration_end")
    _, iteration, losses, metrics = iteration_end_call
    assert metrics == [validation_evaluation]


def test_logger_and_checkpoint_receive_iteration_end_with_the_right_iteration():
    strategy = _StubStrategy(_evaluation())
    logger = _SpyLogger()
    checkpoint = _SpyCheckpoint()
    trainer = Trainer(
        model=_FakeModel(), objectives=[], strategies=[strategy],
        logger=logger, checkpoint=checkpoint,
    )

    trainer.fit(num_iterations=2)

    logger_iterations = [c[1] for c in logger.calls if c[0] == "iteration_end"]
    checkpoint_iterations = [c[1] for c in checkpoint.calls if c[0] == "iteration_end"]
    assert logger_iterations == [1, 2]
    assert checkpoint_iterations == [1, 2]


def test_checkpoint_receives_train_end_with_final_iteration():
    strategy = _StubStrategy(_evaluation())
    checkpoint = _SpyCheckpoint()
    trainer = Trainer(
        model=_FakeModel(), objectives=[], strategies=[strategy], checkpoint=checkpoint,
    )

    trainer.fit(num_iterations=3)

    assert ("train_end", 3) in checkpoint.calls

def test_exception_from_a_strategy_is_reraised_with_its_original_type():
    strategy = _StubStrategy(RuntimeError("boom"))
    trainer = Trainer(model=_FakeModel(), objectives=[], strategies=[strategy])

    with pytest.raises(RuntimeError, match="boom"):
        trainer.fit(num_iterations=1)


def test_exception_preserves_the_original_traceback():
    """A bare `raise` (rather than raising a new/formatted exception)
    keeps the original traceback, so the failure can still be traced
    back to the frame that actually caused it (`_StubStrategy.step`)."""
    strategy = _StubStrategy(RuntimeError("boom"))
    trainer = Trainer(model=_FakeModel(), objectives=[], strategies=[strategy])

    with pytest.raises(RuntimeError) as excinfo:
        trainer.fit(num_iterations=1)

    assert any(frame.name == "step" for frame in excinfo.traceback)


def test_callbacks_and_checkpoint_are_notified_before_reraising():
    callback = _SpyCallback()
    checkpoint = _SpyCheckpoint()
    strategy = _StubStrategy(RuntimeError("boom"))
    trainer = Trainer(
        model=_FakeModel(), objectives=[], strategies=[strategy],
        checkpoint=checkpoint, callbacks=[callback],
    )

    with pytest.raises(RuntimeError, match="boom"):
        trainer.fit(num_iterations=1)

    assert "exception" in callback.events
    assert ("exception", 1) in checkpoint.calls


def test_verbose_mode_does_not_raise():
    strategy = _StubStrategy(_evaluation())
    trainer = Trainer(model=_FakeModel(), objectives=[], strategies=[strategy])

    trainer.fit(num_iterations=1, verbose=True)  # should not raise


def test_verbose_mode_prints_header_and_row(capsys):
    strategy = _StubStrategy(_evaluation(name="residual", value=2.0, weight=1.0))
    trainer = Trainer(model=_FakeModel(), objectives=[], strategies=[strategy])

    trainer.fit(num_iterations=1, verbose=True)

    output = capsys.readouterr().out
    assert "residual" in output
    assert "loss" in output


def test_logger_only_receives_iteration_end_not_the_rest_of_the_lifecycle():
    """
    NOTE: unlike `checkpoint` (which receives `on_train_end`/
    `on_exception`), `logger` currently only ever receives
    `on_iteration_end`. `on_train_start`, `on_iteration_start`,
    `on_train_end`, and `on_exception` are never called on it. This is
    not a crash — Logger's hooks for those events are harmless no-ops
    today — but it's an inconsistency worth being aware of if `Logger`
    ever needs to do something at those points (e.g. flush or close a
    file). This test documents the current, still-open gap.
    """
    strategy = _StubStrategy(_evaluation())
    logger = _SpyLogger()
    checkpoint = _SpyCheckpoint()
    trainer = Trainer(
        model=_FakeModel(), objectives=[], strategies=[strategy],
        logger=logger, checkpoint=checkpoint,
    )

    trainer.fit(num_iterations=1)

    logger_events = {call[0] for call in logger.calls}
    checkpoint_events = {call[0] for call in checkpoint.calls}
    assert logger_events == {"iteration_end"}
    assert "train_end" in checkpoint_events
