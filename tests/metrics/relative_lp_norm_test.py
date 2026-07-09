import pytest
import torch

from sciml.metrics.relative_lp_norm import RelativeLpNorm


def test_relative_lp_norm_evaluates_relative_error_and_name():
    predictions = torch.tensor([1.0, 2.0, 4.0])
    targets = torch.tensor([1.0, 0.0, 1.0])

    metric = RelativeLpNorm(p=2, eps=0.0)
    expected = torch.sqrt(torch.tensor(13.0)) / torch.sqrt(torch.tensor(2.0))

    assert metric.name == "relative_l2"
    assert torch.isclose(metric.evaluate(predictions, targets), expected)


def test_relative_lp_norm_rejects_invalid_order():
    with pytest.raises(ValueError):
        RelativeLpNorm(p=0)
