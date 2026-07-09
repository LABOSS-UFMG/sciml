import torch

from sciml.metrics.mean_absolute_error import MeanAbsoluteError


def test_mean_absolute_error_evaluates_mean_absolute_error_and_name():
    predictions = torch.tensor([1.0, 2.0, 4.0])
    targets = torch.tensor([1.0, 0.0, 1.0])

    metric = MeanAbsoluteError()

    assert metric.name == "mae"
    assert torch.isclose(metric.evaluate(predictions, targets), torch.tensor(5.0 / 3.0))
