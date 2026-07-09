import pytest
import torch

from sciml.metrics.lp_norm import LpNorm


def test_lp_norm_evaluates_error_norm_and_name():
    predictions = torch.tensor([1.0, 2.0, 4.0])
    targets = torch.tensor([1.0, 0.0, 1.0])

    metric = LpNorm(p=2)

    assert metric.name == "l2"
    assert torch.isclose(metric.evaluate(predictions, targets), torch.sqrt(torch.tensor(13.0)))


def test_lp_norm_rejects_invalid_order():
    with pytest.raises(ValueError):
        LpNorm(p=0)
