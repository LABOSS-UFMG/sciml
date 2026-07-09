import torch

from sciml.metrics.mean_squared_error import MeanSquaredError


def test_mean_squared_error_evaluates_mean_squared_error_and_name():
    predictions = torch.tensor([1.0, 2.0, 4.0])
    targets = torch.tensor([1.0, 0.0, 1.0])

    metric = MeanSquaredError()

    assert metric.name == "mse"
    assert torch.isclose(metric.evaluate(predictions, targets), torch.tensor(13.0 / 3.0))
