#####################################################################################
import torch

from abc import abstractmethod
from typing import Dict, Any

from sciml.core.loss import LossBase
from sciml.core.context import Context
from sciml.core.metric import MetricBase
from sciml.metrics import MeanSquaredError
from sciml.utils import checker

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
    ...     def residual(self, context):
    ...         x = context["x"]
    ...         t = context["t"]
    ...         u = context["u"]
    ...
    ...         u_xx = context.partial(u, x, order=2)
    ...         u_t = context.partial(u, t)
    ...
    ...         return u_t - self.alpha * u_xx
    >>>
    >>> loss = HeatEquationLoss(
    ...     alpha=0.1, name="heat_residual", weight=1.0,
    ... )
    """

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
        checker.is_string(name)
        checker.is_float(weight)
        checker.is_dtype(reduction, MetricBase)

        # ---------------------------------------------------------------------------
        # > Inputs
        self.name = name
        self.weight = weight
        self.reduction = reduction

        # ---------------------------------------------------------------------------
        return

    @abstractmethod
    def residual(self, context: Context) -> torch.Tensor:
        """
        Compute the PDE residual based on the batch.

        This is the only method a subclass needs to implement: given the
        network and the batch, return a tensor that should equal zero when
        the differential equation is satisfied.

        Parameters
        ----------
        context : Context
            Evaluation context produced by the objective.

        Returns
        -------
        torch.Tensor
            The residual of the differential equation at each point.
        """
        pass

    def evaluate(self, context: Context) -> torch.Tensor:
        """
        Compute the residual loss for this batch of sampled points.

        Parameters
        ----------
        context : Context
            Evaluation context produced by the objective.

        Returns
        -------
        torch.Tensor
            Scalar tensor representing the loss.
        """
        # ---------------------------------------------------------------------------
        predictions = self.residual(context)
        targets = torch.zeros_like(predictions)

        # ---------------------------------------------------------------------------
        return self.reduction.evaluate(predictions, targets)

#####################################################################################
