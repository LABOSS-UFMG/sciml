#####################################################################################
import torch

from abc import ABC, abstractmethod

from sciml.core.contracts import Batch

#####################################################################################
class LossBase(ABC):
    """
    Abstract base class for defining loss terms used in PINN training.

    Physics-Informed Neural Networks (PINNs) typically combine several loss
    components computed from different sets of sampled points. Common
    examples include:

    - collocation loss: enforces the PDE residual at interior points;
    - boundary loss: enforces boundary conditions;
    - initial loss: enforces initial conditions (for time-dependent problems);
    - observational loss: fits data measurements when available.

    Each of these is implemented as a separate ``LossBase`` subclass.
    """
    # Subclasses should override:
    name: str       # Short identifier used when logging or plotting this loss.
    weight: float   # Scalar weight applied to this loss when combined with others.

    @abstractmethod
    def evaluate(self, network: torch.nn.Module, batch: Batch) -> torch.Tensor:
        """
        Compute the loss value for a batch of sampled points.

        Parameters
        ----------
        network : torch.nn.Module
            Neural network being trained.
        batch : Batch
            Sampled points and any associated data required to evaluate this
            loss (see ``sciml.core.contracts.Batch``).

        Returns
        -------
        torch.Tensor
            Scalar tensor representing the loss.
        """
        pass

#####################################################################################