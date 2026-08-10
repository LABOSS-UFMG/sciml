# -------------------------------------------------------------------------------- #
import torch

from sciml.interfaces.metric import MetricBase

# -------------------------------------------------------------------------------- #
class LpNorm(MetricBase):
    """
    Lp norm of the prediction error.

    Computes the (unnormalized) Lp norm of the difference between
    predictions and targets, using ``torch.norm`` as the underlying
    implementation.

    Examples
    --------
    >>> metric = LpNorm(p=2)
    >>> metric.evaluate(predictions, targets)
    """

    def __init__(self, p: int = 2) -> None:
        """
        Parameters
        ----------
        p : int, default=2
            Order of the norm. Must satisfy ``p >= 1``.
        """
        # ------------------------------------------------------------------------ #
        super().__init__(f"l{p}")

        if p < 1:
            raise ValueError("p must satisfy p >= 1")

        self.p = p
        # ------------------------------------------------------------------------ #
        return

    def evaluate(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return torch.norm(y_pred - y_true, p=self.p)

# -------------------------------------------------------------------------------- #
