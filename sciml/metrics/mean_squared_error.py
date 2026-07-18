#####################################################################################
import torch

from sciml.core.metric import MetricBase

#####################################################################################
class MeanSquaredError(MetricBase):
    """
    Mean Squared Error (MSE) metric.

    Computes the average squared difference between predictions and
    targets.

    Examples
    --------
    >>> metric = MeanSquaredError()
    >>> metric.evaluate(predictions, targets)
    """
    
    def __init__(self):
        super().__init__("mse")
        return

    def evaluate(
            self,
            predictions: torch.Tensor,
            targets: torch.Tensor,
        ) -> torch.Tensor:
        """
        Compute the mean squared error.

        Parameters
        ----------
        predictions : torch.Tensor
            Values predicted by the network.
        targets : torch.Tensor
            Reference values that predictions are compared against.

        Returns
        -------
        torch.Tensor
            Scalar tensor with the mean squared error.
        """
        # ---------------------------------------------------------------------------
        diff = predictions  - targets

        # ---------------------------------------------------------------------------
        return torch.mean(diff * diff)

#####################################################################################
