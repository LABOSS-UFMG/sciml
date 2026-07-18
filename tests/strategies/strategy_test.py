import pytest
import torch

from sciml.strategies.strategy import Strategy
from sciml.core.model import ModelBase
from sciml.core.evaluation import Evaluation


class _FakeModel(ModelBase):
    """A model wrapping a single trainable scalar parameter."""

    def __init__(self, initial: float):
        self.param = torch.nn.Parameter(torch.tensor(float(initial)))

    def compute(self, context):
        return


class _ObjectiveStub:
    """Returns a fixed Evaluation (or None) regardless of the model,
    while recording every call for later assertions."""

    def __init__(self, evaluation):
        self._evaluation = evaluation
        self.received = []

    def evaluate(self, model, keys):
        self.received.append((model, keys))
        return self._evaluation


def _make_optimizer(model: _FakeModel) -> torch.optim.Optimizer:
    return torch.optim.SGD([model.param], lr=0.1)


def test_rejects_non_optimizer():
    with pytest.raises(TypeError):
        Strategy(optimizer="not an optimizer")


def test_accepts_a_real_optimizer():
    model = _FakeModel(1.0)
    strategy = Strategy(optimizer=_make_optimizer(model))

    assert strategy.optimizer is not None


def test_name_defaults_to_none():
    model = _FakeModel(1.0)
    strategy = Strategy(optimizer=_make_optimizer(model))

    assert strategy.name is None


def test_name_must_be_a_string_when_given():
    model = _FakeModel(1.0)
    with pytest.raises(TypeError):
        Strategy(optimizer=_make_optimizer(model), name=123)


def test_keys_default_to_none():
    model = _FakeModel(1.0)
    strategy = Strategy(optimizer=_make_optimizer(model))

    assert strategy.keys is None


def test_keys_must_be_an_iterable_of_strings():
    model = _FakeModel(1.0)
    with pytest.raises(TypeError):
        Strategy(optimizer=_make_optimizer(model), keys=[1, 2])


def test_evaluations_start_empty():
    model = _FakeModel(1.0)
    strategy = Strategy(optimizer=_make_optimizer(model))

    assert strategy._evaluations == []


def test_evaluate_collects_results_from_every_objective():
    model = _FakeModel(1.0)
    strategy = Strategy(optimizer=_make_optimizer(model))

    evaluation_a = Evaluation(values={"a": torch.tensor(1.0)}, weights={"a": 1.0}, metadata={})
    evaluation_b = Evaluation(values={"b": torch.tensor(2.0)}, weights={"b": 1.0}, metadata={})
    objective_a = _ObjectiveStub(evaluation_a)
    objective_b = _ObjectiveStub(evaluation_b)

    result = strategy.evaluate(model, [objective_a, objective_b])

    assert result == [evaluation_a, evaluation_b]


def test_evaluate_skips_objectives_that_return_none():
    model = _FakeModel(1.0)
    strategy = Strategy(optimizer=_make_optimizer(model))

    evaluation = Evaluation(values={"a": torch.tensor(1.0)}, weights={"a": 1.0}, metadata={})
    active_objective = _ObjectiveStub(evaluation)
    inactive_objective = _ObjectiveStub(None)

    result = strategy.evaluate(model, [active_objective, inactive_objective])

    assert result == [evaluation]


def test_evaluate_passes_self_keys_to_each_objective():
    model = _FakeModel(1.0)
    strategy = Strategy(optimizer=_make_optimizer(model), keys=["residual"])
    # `values`/`weights` are now required fields (no shared mutable
    # default), so an "empty" Evaluation must be built explicitly.
    objective = _ObjectiveStub(Evaluation(values={}, weights={}, metadata={}))

    strategy.evaluate(model, [objective])

    assert objective.received == [(model, ["residual"])]


def test_evaluate_with_no_objectives_returns_empty_list():
    model = _FakeModel(1.0)
    strategy = Strategy(optimizer=_make_optimizer(model))

    result = strategy.evaluate(model, [])

    assert result == []


def test_objective_function_with_no_evaluations_is_zero():
    model = _FakeModel(1.0)
    strategy = Strategy(optimizer=_make_optimizer(model))

    result = strategy.objective_function([])

    assert result == 0.0


def test_objective_function_combines_a_single_evaluation():
    model = _FakeModel(1.0)
    strategy = Strategy(optimizer=_make_optimizer(model))
    evaluation = Evaluation(
        values={"residual": torch.tensor(2.0)}, weights={"residual": 3.0},
        metadata={},
    )

    result = strategy.objective_function([evaluation])

    assert torch.isclose(result, torch.tensor(6.0))


def test_objective_function_sums_multiple_keys_within_one_evaluation():
    model = _FakeModel(1.0)
    strategy = Strategy(optimizer=_make_optimizer(model))
    evaluation = Evaluation(
        values={"a": torch.tensor(1.0), "b": torch.tensor(2.0)},
        weights={"a": 1.0, "b": 10.0},
        metadata={},
    )
    # 1*1 + 2*10 = 21

    result = strategy.objective_function([evaluation])

    assert torch.isclose(result, torch.tensor(21.0))


def test_objective_function_sums_across_multiple_evaluations():
    model = _FakeModel(1.0)
    strategy = Strategy(optimizer=_make_optimizer(model))
    evaluation_a = Evaluation(values={"a": torch.tensor(1.0)}, weights={"a": 1.0}, metadata={})
    evaluation_b = Evaluation(values={"b": torch.tensor(2.0)}, weights={"b": 1.0}, metadata={})

    result = strategy.objective_function([evaluation_a, evaluation_b])

    assert torch.isclose(result, torch.tensor(3.0))


def test_closure_zeroes_gradients_before_evaluating():
    model = _FakeModel(2.0)
    optimizer = _make_optimizer(model)
    strategy = Strategy(optimizer=optimizer)
    objective = _ObjectiveStub(
        Evaluation(values={"a": model.param ** 2}, weights={"a": 1.0}, metadata={})
    )

    # Simulate a leftover gradient from a previous, unrelated step.
    model.param.grad = torch.tensor(999.0)

    strategy.closure(model, [objective])

    # The stale gradient must have been replaced, not accumulated onto.
    assert model.param.grad is not None
    assert model.param.grad.item() != 999.0


def test_closure_computes_gradients_via_backward():
    model = _FakeModel(3.0)
    optimizer = _make_optimizer(model)
    strategy = Strategy(optimizer=optimizer)

    def make_objective():
        # d(param^2)/d(param) = 2*param = 6.0 at param=3.0
        return _ObjectiveStub(
            Evaluation(values={"a": model.param ** 2}, weights={"a": 1.0}, metadata={})
        )

    strategy.closure(model, [make_objective()])

    assert torch.isclose(model.param.grad, torch.tensor(6.0))


def test_closure_returns_the_objective_function_value():
    model = _FakeModel(2.0)
    optimizer = _make_optimizer(model)
    strategy = Strategy(optimizer=optimizer)
    objective = _ObjectiveStub(
        Evaluation(values={"a": model.param ** 2}, weights={"a": 1.0}, metadata={})
    )

    result = strategy.closure(model, [objective])

    assert torch.isclose(result, torch.tensor(4.0))


def test_closure_stores_evaluations_for_later_use_by_step():
    model = _FakeModel(2.0)
    optimizer = _make_optimizer(model)
    strategy = Strategy(optimizer=optimizer)
    evaluation = Evaluation(values={"a": model.param ** 2}, weights={"a": 1.0}, metadata={})
    objective = _ObjectiveStub(evaluation)

    strategy.closure(model, [objective])

    assert strategy._evaluations == [evaluation]


def test_step_updates_the_model_parameter():
    model = _FakeModel(2.0)
    optimizer = _make_optimizer(model)
    strategy = Strategy(optimizer=optimizer)
    objective = _ObjectiveStub(
        Evaluation(values={"a": model.param ** 2}, weights={"a": 1.0}, metadata={})
    )
    original_value = model.param.item()

    strategy.step(model, [objective])

    assert model.param.item() != original_value


def test_step_returns_an_evaluation_with_combined_values_and_weights():
    model = _FakeModel(2.0)
    optimizer = _make_optimizer(model)
    strategy = Strategy(optimizer=optimizer)
    objective = _ObjectiveStub(
        Evaluation(values={"a": model.param ** 2}, weights={"a": 1.0}, metadata={})
    )

    result = strategy.step(model, [objective])

    assert result.values == {"a": pytest.approx(4.0)}
    assert result.weights == {"a": 1.0}


def test_step_metadata_defaults_to_training_name():
    model = _FakeModel(2.0)
    optimizer = _make_optimizer(model)
    strategy = Strategy(optimizer=optimizer)
    objective = _ObjectiveStub(
        Evaluation(values={"a": model.param ** 2}, weights={"a": 1.0}, metadata={})
    )

    result = strategy.step(model, [objective])

    assert result.metadata == {"name": "training"}


def test_step_metadata_uses_the_given_name():
    model = _FakeModel(2.0)
    optimizer = _make_optimizer(model)
    strategy = Strategy(optimizer=optimizer, name="adam_phase")
    objective = _ObjectiveStub(
        Evaluation(values={"a": model.param ** 2}, weights={"a": 1.0}, metadata={})
    )

    result = strategy.step(model, [objective])

    assert result.metadata == {"name": "adam_phase"}


def test_step_uses_the_evaluations_produced_by_its_own_closure_call():
    """`step` must reflect exactly what its own closure computed during
    the optimizer's call, not some stale or unrelated state."""
    model = _FakeModel(5.0)
    optimizer = _make_optimizer(model)
    strategy = Strategy(optimizer=optimizer)

    class _RecomputingObjective:
        """Recomputes its value from the model's *current* parameter
        each time it's evaluated, mimicking a real Objective/Loss."""
        def evaluate(self, model, keys):
            return Evaluation(values={"a": model.param ** 2}, weights={"a": 1.0}, metadata={})

    result = strategy.step(model, [_RecomputingObjective()])

    assert result.values["a"] == pytest.approx(25.0)
