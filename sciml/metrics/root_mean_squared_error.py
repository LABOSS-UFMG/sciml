#####################################################################################
import torch

from sciml.core.metric import MetricBase

#####################################################################################
class RootMeanSquaredError(MetricBase):
    """
    Root Mean Squared Error (RMSE) metric.

    Computes the square root of the average squared difference between
    predictions and targets.

    Examples
    --------
    >>> metric = RootMeanSquaredError()
    >>> metric.evaluate(predictions, targets)
    """

    name = "rmse"

    def evaluate(
            self,
            predictions: torch.Tensor,
            targets: torch.Tensor,
        ) -> torch.Tensor:
        """
        Compute the root mean squared error.

        Parameters
        ----------
        predictions : torch.Tensor
            Values predicted by the network.
        targets : torch.Tensor
            Reference values that predictions are compared against.

        Returns
        -------
        torch.Tensor
            Scalar tensor with the root mean squared error.
        """
        # ---------------------------------------------------------------------------
        return torch.sqrt(torch.mean((predictions - targets) ** 2))

#####################################################################################
