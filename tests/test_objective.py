# -------------------------------------------------------------------------------- #
import pytest

from sciml.core.objective import Objective
from conftest import LinearModel, SquaredLoss, ConstantSampler

# -------------------------------------------------------------------------------- #
def test_evaluate_combines_weighted_losses():
    model = LinearModel()
    loss_a = SquaredLoss(name="a", weight=2.0)
    loss_b = SquaredLoss(name="b", weight=3.0)

    objective = Objective(name="obj", sampler=ConstantSampler(), losses=[loss_a, loss_b])
    evaluation = objective.evaluate(model)

    assert evaluation.name == "obj"
    assert set(evaluation.losses) == {"a", "b"}
    assert evaluation.weights == {"a": 2.0, "b": 3.0}

    expected = 2.0 * evaluation.losses["a"] + 3.0 * evaluation.losses["b"]
    assert evaluation.objective.item() == pytest.approx(expected, rel=1e-5)

# -------------------------------------------------------------------------------- #
