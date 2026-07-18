#####################################################################################
import torch

from sciml.core.metric import MetricBase

#####################################################################################
class MeanAbsoluteError(MetricBase):
    """
    Mean Absolute Error (MAE) metric.

    Computes the average absolute difference between predictions and
    targets.

    Examples
    --------
    >>> metric = MeanAbsoluteError()
    >>> metric.evaluate(predictions, targets)
    """
    
    def __init__(self):
        super().__init__("mae")
        return

    def evaluate(
            self,
            predictions: torch.Tensor,
            targets: torch.Tensor,
        ) -> torch.Tensor:
        """
        Compute the mean absolute error.

        Parameters
        ----------
        predictions : torch.Tensor
            Values predicted by the network.
        targets : torch.Tensor
            Reference values that predictions are compared against.

        Returns
        -------
        torch.Tensor
            Scalar tensor with the mean absolute error.
        """
        # ---------------------------------------------------------------------------
        return torch.mean(torch.abs(predictions - targets))

#####################################################################################
