import torch

from sciml.metrics.root_mean_squared_error import RootMeanSquaredError


def test_root_mean_squared_error_evaluates_rmse_and_name():
    predictions = torch.tensor([1.0, 2.0, 4.0])
    targets = torch.tensor([1.0, 0.0, 1.0])

    metric = RootMeanSquaredError()

    assert metric.name == "rmse"
    assert torch.isclose(metric.evaluate(predictions, targets), torch.sqrt(torch.tensor(13.0 / 3.0)))
