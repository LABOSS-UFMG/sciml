#####################################################################################
import torch

from abc import abstractmethod
from typing import Dict, Any

from sciml.core.loss import LossBase
from sciml.core.contracts import Batch
from sciml.core.metric import MetricBase
from sciml.metrics import MeanSquaredError
from sciml.utils import validation
from sciml.utils.autograd import derivative

#####################################################################################
class Residual(LossBase):
    """
    Abstract helper base class for PDE-residual losses.

    A residual loss enforces that some differential equation is satisfied
    by the network's output at a set of sampled points (e.g. collocation
    points in the interior of the domain). This class handles the common
    boilerplate shared by every residual loss — extracting the sampled
    inputs, enabling autograd on them, and reducing the residual to a
    scalar loss — so that a subclass only needs to implement the
    problem-specific residual formula.

    Note
    ----
    Unlike ``SupervisedLoss``, this class is still abstract: it is meant
    to be subclassed once per differential equation, not instantiated
    directly. The convenience method ``self.derivative(...)`` is available
    to every subclass without any additional import (it is an alias for
    ``sciml.utils.autograd.derivative``).

    Examples
    --------
    A residual loss for the 1D heat equation ``u_t = alpha * u_xx``:

    >>> class HeatEquationLoss(Residual):
    ...     def __init__(self, alpha: float, **kwargs):
    ...         super().__init__(**kwargs)
    ...         self.alpha = alpha
    ...
    ...     def residual(self, network, x):
    ...         u = network(x)
    ...         grad_u = self.derivative(u, x)          # [u_x, u_t]
    ...         u_x, u_t = grad_u[:, 0:1], grad_u[:, 1:2]
    ...         u_xx = self.derivative(u_x, x)[:, 0:1]
    ...         return u_t - self.alpha * u_xx
    >>>
    >>> loss = HeatEquationLoss(
    ...     alpha=0.1, name="heat_residual", weight=1.0, input_key="xt",
    ... )
    """

    # Convenience alias so subclasses can call `self.derivative(...)`
    # without an extra import. See `sciml.utils.autograd.derivative` for
    # the full implementation and docstring.
    derivative = staticmethod(derivative)

    def __init__(
            self,
            name: str,
            weight: float,
            reduction: MetricBase = MeanSquaredError(),
        ) -> None:
        """
        Parameters
        ----------
        name : str
            Short identifier used when logging or plotting this loss
            (e.g. ``"heat_residual"``).
        weight : float
            Scalar weight applied to this loss when combined with others.
        reduction : MetricBase, optional
            Metric used to reduce the residual (compared against zero) to
            a scalar loss. Defaults to ``MeanSquaredError()``.
        """
        # ---------------------------------------------------------------------------
        # > Validation
        validation.is_string(name)
        validation.is_float(weight)
        validation.is_metric(reduction)

        # ---------------------------------------------------------------------------
        # > Inputs
        self.name = name
        self.weight = weight
        self.reduction = reduction

        # ---------------------------------------------------------------------------
        return

    @abstractmethod
    def residual(self, network: torch.nn.Module, data: Dict[str, Any]) -> torch.Tensor:
        """
        Compute the PDE residual based on the batch.

        This is the only method a subclass needs to implement: given the
        network and the batch, return a tensor that should equal zero when
        the differential equation is satisfied.

        Parameters
        ----------
        network : torch.nn.Module
            Neural network being trained.
        data : Dict[str, Any]
            Data of batch.inputs.

        Returns
        -------
        torch.Tensor
            The residual of the differential equation at each point.
        """
        pass

    def evaluate(self, network: torch.nn.Module, batch: Batch) -> torch.Tensor:
        """
        Compute the residual loss for this batch of sampled points.

        Parameters
        ----------
        network : torch.nn.Module
            Neural network being trained.
        batch : Batch
            Must contain ``self.input_key`` in ``batch.inputs``.

        Returns
        -------
        torch.Tensor
            Scalar tensor representing the loss.
        """
        # ---------------------------------------------------------------------------
        residual = self.residual(network, batch.inputs)
        target = torch.zeros_like(residual)

        # ---------------------------------------------------------------------------
        return self.reduction.evaluate(residual, target)

#####################################################################################
