#####################################################################################
import torch

from sciml.core.metric import MetricBase

#####################################################################################
class MaxError(MetricBase):
    """
    MAximum error or L_infity.

    Computes the maximum difference between the values of two vectors.

    Examples
    --------
    >>> metric = MaxError()
    >>> metric.evaluate(predictions, targets)
    """
    name = "max_error"

    def evaluate(
            self,
            predictions: torch.Tensor,
            targets: torch.Tensor,
        ) -> torch.Tensor:
        """
        Compute the maximum difference between the values of two vectors.

        Parameters
        ----------
        predictions : torch.Tensor
            Values predicted by the network.
        targets : torch.Tensor
            Reference values that predictions are compared against.

        Returns
        -------
        torch.Tensor
            Scalar tensor with the maximum difference.
        """
        # ---------------------------------------------------------------------------
        return torch.max(torch.abs(predictions - targets))

#####################################################################################
