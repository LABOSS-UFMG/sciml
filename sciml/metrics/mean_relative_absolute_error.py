#####################################################################################
import torch

from sciml.core.metric import MetricBase

#####################################################################################
class MeanRelativeAbsoluteError(MetricBase):
    """
    Mean Relative Absolute Error (MRAE) metric.

    Computes the average relative absolute difference between predictions
    and targets.

    The relative error is computed using the absolute value of the target
    in the denominator. A small epsilon is added to avoid division by zero.

    Examples
    --------
    >>> metric = MeanRelativeAbsoluteError()
    >>> metric.evaluate(predictions, targets)
    """

    def __init__(
            self,
            eps: float = 1e-8,
        ) -> None:
        """
        Parameters
        ----------
        eps : float, default=1e-8
            Small value added to the denominator to avoid division by zero.
        """
        super().__init__("mrae")
        self.eps = eps
        return

    def evaluate(
            self,
            predictions: torch.Tensor,
            targets: torch.Tensor,
        ) -> torch.Tensor:
        """
        Compute the mean relative absolute error.

        Parameters
        ----------
        predictions : torch.Tensor
            Values predicted by the network.
        targets : torch.Tensor
            Reference values that predictions are compared against.

        Returns
        -------
        torch.Tensor
            Scalar tensor with the mean relative absolute error.
        """
        # ---------------------------------------------------------------------------
        numerator = torch.abs(predictions - targets)
        denominator = torch.abs(targets) + self.eps

        # ---------------------------------------------------------------------------
        return torch.mean(numerator / denominator)

#####################################################################################
