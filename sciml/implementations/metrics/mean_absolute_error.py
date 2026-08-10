# -------------------------------------------------------------------------------- #
import torch

from sciml.interfaces.metric import MetricBase

# -------------------------------------------------------------------------------- #
class MeanAbsoluteError(MetricBase):
    """
    Mean Absolute Error (MAE) metric.

    Computes the average absolute difference between predictions and
    targets, delegating the computation to ``torch.nn.functional``.

    Examples
    --------
    >>> metric = MeanAbsoluteError()
    >>> metric.evaluate(predictions, targets)
    """

    def __init__(self) -> None:
        # ------------------------------------------------------------------------ #
        super().__init__("mae")
        # ------------------------------------------------------------------------ #
        return

    def evaluate(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.l1_loss(y_pred, y_true)

# -------------------------------------------------------------------------------- #
