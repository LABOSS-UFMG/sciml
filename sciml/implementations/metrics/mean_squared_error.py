# -------------------------------------------------------------------------------- #
import torch

from sciml.interfaces.metric import MetricBase

# -------------------------------------------------------------------------------- #
class MeanSquaredError(MetricBase):
    """
    Mean Squared Error (MSE) metric.

    Computes the average squared difference between predictions and
    targets, delegating the computation to ``torch.nn.functional``.

    Examples
    --------
    >>> metric = MeanSquaredError()
    >>> metric.evaluate(predictions, targets)
    """

    def __init__(self) -> None:
        # ------------------------------------------------------------------------ #
        super().__init__("mse")
        # ------------------------------------------------------------------------ #
        return

    def evaluate(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.mse_loss(y_pred, y_true)

# -------------------------------------------------------------------------------- #
