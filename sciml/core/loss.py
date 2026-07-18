#####################################################################################
import torch

from abc import ABC, abstractmethod

from sciml.core.context import Context

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
    def __init__(self):
        """
        Parameters
        ----------
        name : str
            Short identifier used when logging or plotting this loss.
        weight : float
            Scalar weight applied to this loss when combined with others.
        """
        # ---------------------------------------------------------------------------
        self.name: str
        self.weight: float
        # ---------------------------------------------------------------------------
        return

    @abstractmethod
    def evaluate(
            self,
            context: Context,
        ) -> torch.Tensor:
        """
        Compute the loss value for a batch of sampled points.

        Parameters
        ----------
        context : Context
            Evaluation context produced by the objective.

        Returns
        -------
        torch.Tensor
            Scalar tensor representing the loss.
        """
        pass

#####################################################################################