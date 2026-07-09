import torch

from sciml.metrics.max_error import MaxError


def test_max_error_evaluates_maximum_absolute_error_and_name():
    predictions = torch.tensor([1.0, 2.0, 4.0])
    targets = torch.tensor([1.0, 0.0, 1.0])

    metric = MaxError()

    assert metric.name == "max_error"
    assert torch.equal(metric.evaluate(predictions, targets), torch.tensor(3.0))
