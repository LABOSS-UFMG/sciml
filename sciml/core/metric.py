#####################################################################################
import torch

from abc import ABC, abstractmethod

#####################################################################################
class MetricBase(ABC):
    """
    Abstract base class for defining evaluation metrics.

    A metric quantifies how well the network's predictions match a reference
    (e.g. an analytical solution or reference data). Unlike a loss, a metric
    is only monitored during training/evaluation — it is not used to compute
    gradients or update the network's parameters.
    """
    # Subclasses should override:
    name: str       # Name of the metric 

    @abstractmethod
    def evaluate(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute the metric value.

        Parameters
        ----------
        predictions : torch.Tensor
            Values predicted by the network.
        targets : torch.Tensor
            Reference values that predictions are compared against
            (e.g. analytical solution or measured data).

        Returns
        -------
        torch.Tensor
            A tensor representing the metric value.
        """
        pass
    
#####################################################################################
