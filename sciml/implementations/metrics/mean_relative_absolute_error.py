# -------------------------------------------------------------------------------- #
import torch

from sciml.interfaces.metric import MetricBase

# -------------------------------------------------------------------------------- #
class MeanRelativeAbsoluteError(MetricBase):
    """
    Mean Relative Absolute Error (MRAE) metric.

    Computes the average relative absolute difference between predictions
    and targets. The relative error uses the absolute value of the target
    in the denominator, with a small epsilon to avoid division by zero.
    There is no dedicated ``torch.nn`` loss for this reduction, so it is
    built directly from ``torch.abs``/``torch.mean``.

    Examples
    --------
    >>> metric = MeanRelativeAbsoluteError()
    >>> metric.evaluate(predictions, targets)
    """

    def __init__(self, eps: float = 1e-8) -> None:
        """
        Parameters
        ----------
        eps : float, default=1e-8
            Small value added to the denominator to avoid division by zero.
        """
        # ------------------------------------------------------------------------ #
        super().__init__("mrae")

        self.eps = eps
        # ------------------------------------------------------------------------ #
        return

    def evaluate(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        # ------------------------------------------------------------------------ #
        numerator = torch.abs(y_pred - y_true)
        denominator = torch.abs(y_true) + self.eps

        # ------------------------------------------------------------------------ #
        return torch.mean(numerator / denominator)

# -------------------------------------------------------------------------------- #
