# -------------------------------------------------------------------------------- #
import torch

from sciml.interfaces.metric import MetricBase

# -------------------------------------------------------------------------------- #
class RelativeLpNorm(MetricBase):
    """
    Relative Lp norm of the prediction error.

    Computes the Lp norm of the difference between predictions and targets,
    normalized by the Lp norm of the targets, using ``torch.norm`` as the
    underlying implementation.

    Examples
    --------
    >>> metric = RelativeLpNorm(p=2)
    >>> metric.evaluate(predictions, targets)
    """

    def __init__(self, p: int = 2, eps: float = 1e-12) -> None:
        """
        Parameters
        ----------
        p : int, default=2
            Order of the norm. Must satisfy ``p >= 1``.
        eps : float, default=1e-12
            Small constant added to the denominator to avoid division by
            zero when ``targets`` is (or is close to) the zero tensor.
        """
        # ------------------------------------------------------------------------ #
        super().__init__(f"relative_l{p}")

        if p < 1:
            raise ValueError("p must satisfy p >= 1")

        self.p = p
        self.eps = eps
        # ------------------------------------------------------------------------ #
        return

    def evaluate(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        # ------------------------------------------------------------------------ #
        numerator = torch.norm(y_pred - y_true, p=self.p)
        denominator = torch.norm(y_true, p=self.p) + self.eps

        # ------------------------------------------------------------------------ #
        return numerator / denominator

# -------------------------------------------------------------------------------- #
