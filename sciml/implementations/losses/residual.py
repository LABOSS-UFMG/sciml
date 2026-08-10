# -------------------------------------------------------------------------------- #
import torch

from typing import Callable

from sciml.interfaces.loss import LossBase
from sciml.interfaces.metric import MetricBase
from sciml.contracts.context import Context
from sciml.implementations.metrics import MeanSquaredError

# -------------------------------------------------------------------------------- #
class Residual(LossBase):
    """
    Abstract helper base class for PDE-residual losses.

    A residual loss enforces that some differential equation is satisfied
    by the network's output at a set of sampled points (e.g. collocation
    points in the interior of the domain).

    Examples
    --------
    A residual loss for the 1D heat equation ``u_t = alpha * u_xx``:

    >>> def residual(context):
    ...     x = context["x"]
    ...     t = context["t"]
    ...     u = context["u"]
    ...
    ...     u_xx = context.partial(u, x, order=2)
    ...     u_t = context.partial(u, t)
    ...
    ...     return u_t - 0.1 * u_xx
    ...
    >>> loss = Residual(
    ...     name="heat_residual",
    ...     weight=1.0,
    ...     residual=residual,
    ... )
    """

    def __init__(
            self,
            name: str,
            weight: float,
            residual: Callable[[Context], torch.Tensor],
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
        residual : Callable[[Context], torch.Tensor]
            Function that computes the PDE residual based on the batch.
        reduction : MetricBase, optional
            Metric used to reduce the residual (compared against zero) to
            a scalar loss. Defaults to ``MeanSquaredError()``.
        """
        # ------------------------------------------------------------------------ #
        # Store constructor arguments
        self.name = name
        self.weight = weight
        self.residual = residual
        self.reduction = reduction

        # ------------------------------------------------------------------------ #
        return

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
        # ------------------------------------------------------------------------ #
        y_pred = self.residual(context)
        y_true = torch.zeros_like(y_pred)

        # ------------------------------------------------------------------------ #
        return self.reduction.evaluate(y_pred, y_true)

# -------------------------------------------------------------------------------- #
