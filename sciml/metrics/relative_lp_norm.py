#####################################################################################
import torch

from sciml.core.metric import MetricBase
from sciml.utils import checker

#####################################################################################
class RelativeLpNorm(MetricBase):
    """
    Relative Lp norm of the prediction error.

    Computes the Lp norm of the difference between predictions and targets,
    normalized by the Lp norm of the targets:

    Examples
    --------
    >>> metric = RelativeLpNorm(p=2)
    >>> metric.evaluate(predictions, targets)
    """
    
    def __init__(
            self,
            p: int,
            eps: float = 1e-12,
        ) -> None:
        """
        Parameters
        ----------
        p : int
            Order of the norm. Must satisfy ``p >= 1``.
        eps : float, default=1e-12
            Small constant added to the denominator to avoid division by
            zero when ``targets`` is (or is close to) the zero tensor.
        """
        super().__init__("relative_l")
        # ---------------------------------------------------------------------------
        # > Validate
        checker.is_integer(p)
        checker.is_float(eps)

        if p < 1:
            raise ValueError("p must satisfy p >= 1")
        
        # ---------------------------------------------------------------------------
        # > Store
        self.p = p
        self.eps = eps

        # ---------------------------------------------------------------------------
        # > Update name
        self.name += str(p)

        return

    def evaluate(
            self,
            predictions: torch.Tensor,
            targets: torch.Tensor,
        ) -> torch.Tensor:
        """
        Compute the relative Lp norm of the prediction error.

        Parameters
        ----------
        predictions : torch.Tensor
            Values predicted by the network.
        targets : torch.Tensor
            Reference values that predictions are compared against.

        Returns
        -------
        torch.Tensor
            Scalar tensor with the relative Lp norm of the error.
        """
        # ---------------------------------------------------------------------------
        numerator = torch.norm(predictions - targets, p=self.p)
        denominator = torch.norm(targets, p=self.p) + self.eps
        # ---------------------------------------------------------------------------
        return numerator / denominator

#####################################################################################
