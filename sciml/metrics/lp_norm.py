#####################################################################################
import torch

from sciml.core.metric import MetricBase
from sciml.utils import validation

#####################################################################################
class LpNorm(MetricBase):
    """
    Lp norm of the prediction error.

    Computes the (unnormalized) Lp norm of the difference between
    predictions and targets:

    Examples
    --------
    >>> metric = LpNorm(p=2)
    >>> metric.evaluate(predictions, targets)
    """
    name = "l"

    def __init__(self, p: int = 2) -> None:
        """
        Parameters
        ----------
        p : int
            Order of the norm. Must satisfy ``p >= 1``. Default is 2.
        """
        # ---------------------------------------------------------------------------
        # > Validation
        validation.is_integer(p)
        if p < 1:
            raise ValueError("p must satisfy p >= 1")

        # ---------------------------------------------------------------------------
        # > Inputs
        self.p = p

        # ---------------------------------------------------------------------------
        # > Update name
        self.name += str(p)

        # ---------------------------------------------------------------------------
        return

    def evaluate(
            self,
            predictions: torch.Tensor,
            targets: torch.Tensor,
        ) -> torch.Tensor:
        """
        Compute the Lp norm of the prediction error.

        Parameters
        ----------
        predictions : torch.Tensor
            Values predicted by the network.
        targets : torch.Tensor
            Reference values that predictions are compared against.

        Returns
        -------
        torch.Tensor
            Scalar tensor with the Lp norm of the error.
        """
        # ---------------------------------------------------------------------------
        return torch.norm(predictions - targets, p=self.p)

#####################################################################################
