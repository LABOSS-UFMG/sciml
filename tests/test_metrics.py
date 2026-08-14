# -------------------------------------------------------------------------------- #
import math

import pytest
import torch

from sciml.implementations.metrics import (
    LpNorm,
    MaxError,
    MeanAbsoluteError,
    MeanRelativeAbsoluteError,
    MeanSquaredError,
    RelativeLpNorm,
    RootMeanSquaredError,
)

# -------------------------------------------------------------------------------- #
PRED = torch.tensor([1.0, 2.0, 3.0])
TRUE = torch.tensor([1.5, 2.5, 2.5])
# errors = [-0.5, -0.5, 0.5]

@pytest.mark.parametrize(
    "metric, expected",
    [
        (MeanSquaredError(), 0.25),
        (MeanAbsoluteError(), 0.5),
        (RootMeanSquaredError(), 0.5),
        (MaxError(), 0.5),
        (LpNorm(p=2), math.sqrt(3 * 0.25)),
        (RelativeLpNorm(p=2), math.sqrt(3 * 0.25) / torch.norm(TRUE, p=2).item()),
        (MeanRelativeAbsoluteError(), (0.5 / 1.5 + 0.5 / 2.5 + 0.5 / 2.5) / 3),
    ],
)
def test_metric_value(metric, expected):
    result = metric.evaluate(PRED, TRUE).item()

    assert result == pytest.approx(expected, rel=1e-4)

# -------------------------------------------------------------------------------- #
