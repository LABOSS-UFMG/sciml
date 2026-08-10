# -------------------------------------------------------------------------------- #
import torch

from sciml.interfaces.metric import MetricBase

# -------------------------------------------------------------------------------- #
class RootMeanSquaredError(MetricBase):
    """
    Root Mean Squared Error (RMSE) metric.

    Computes the square root of the average squared difference between
    predictions and targets, reusing ``torch.nn.functional.mse_loss`` for
    the underlying reduction.

    Examples
    --------
    >>> metric = RootMeanSquaredError()
    >>> metric.evaluate(predictions, targets)
    """

    def __init__(self) -> None:
        # ------------------------------------------------------------------------ #
        super().__init__("rmse")
        # ------------------------------------------------------------------------ #
        return

    def evaluate(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(torch.nn.functional.mse_loss(y_pred, y_true))

# -------------------------------------------------------------------------------- #
