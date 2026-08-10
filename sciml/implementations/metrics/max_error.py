# -------------------------------------------------------------------------------- #
import torch

from sciml.interfaces.metric import MetricBase

# -------------------------------------------------------------------------------- #
class MaxError(MetricBase):
    """
    Maximum error (L-infinity norm) metric.

    Computes the maximum absolute difference between predictions and
    targets. There is no dedicated ``torch.nn`` loss for this reduction,
    so it is built directly from ``torch.max``/``torch.abs``.

    Examples
    --------
    >>> metric = MaxError()
    >>> metric.evaluate(predictions, targets)
    """

    def __init__(self) -> None:
        # ------------------------------------------------------------------------ #
        super().__init__("max_error")
        # ------------------------------------------------------------------------ #
        return

    def evaluate(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return torch.max(torch.abs(y_pred - y_true))

# -------------------------------------------------------------------------------- #
