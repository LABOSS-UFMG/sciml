#####################################################################################
import torch

from typing import Iterable

from sciml.core.loss import LossBase
from sciml.core.context import Context
from sciml.core.metric import MetricBase
from sciml.metrics import MeanSquaredError
from sciml.utils import checker

#####################################################################################
class Supervised(LossBase):
    """
    Loss term that fits the network's output against known reference
    values at a set of sampled points.

    This single class covers every loss that compares a prediction to a
    known target, regardless of why that target is known:

    - initial condition: known values of the solution at ``t = 0``;
    - boundary condition: known values of the solution at the domain's
      boundary;
    - observational data: known values measured or simulated elsewhere.

    All three cases share the same computation — evaluate the network at
    the sampled inputs and compare the result to the sampled targets — so
    a single, configurable class is used for all of them, instantiated
    once per role (e.g. one instance for the initial condition, another
    for the boundary condition).

    Examples
    --------
    Fitting a known initial condition ``u(x, 0) = sin(pi * x)``:

    >>> initial_loss = Supervised(
    ...     name="initial_condition",
    ...     weight=1.0,
    ...     input_keys=["xt0"],
    ...     target_keys=["u0"],
    ... )
    >>> initial_loss.evaluate(network, batch)
    """

    def __init__(
            self,
            name: str,
            input_keys: Iterable[str],
            target_keys: Iterable[str],
            weight: float = 1.0,
            reduction: MetricBase = MeanSquaredError(),
        ) -> None:
        """
        Parameters
        ----------
        name : str
            Short identifier used when logging or plotting this loss
            (e.g. ``"initial_condition"``, ``"boundary_condition"``,
            ``"observational"``).
        input_keys : Iterable[str]
            Keys used to retrieve the sampled inputs from ``context``
            (e.g. ``"xt0"`` for initial-condition points).
        target_keys : Iterable[str]
            Keys used to retrieve the reference values from
            ``context`` (e.g. ``"u0"`` for the known initial values).
        weight : float
            Scalar weight applied to this loss when combined with others.
        reduction : MetricBase
            Metric used to reduce the difference between the network's
            prediction and the target to a scalar loss. Defaults to
            ``MeanSquaredError()``, which is the standard choice for this
            kind of loss. Any ``MetricBase`` instance can be used instead
            (e.g. ``MeanAbsoluteError()``), since the loss and the metric
            share the same ``evaluate(predictions, targets)`` contract.
        """
        # ---------------------------------------------------------------------------
        # > Validation
        checker.is_string(name)
        checker.is_float(weight)
        checker.is_iterable(input_keys, dtype=str)
        checker.is_iterable(target_keys, dtype=str)
        checker.is_dtype(reduction, MetricBase)

        # ---------------------------------------------------------------------------
        # > Inputs
        self.name = name
        self.weight = weight
        self.input_keys = input_keys
        self.target_keys = target_keys
        self.reduction = reduction

        # ---------------------------------------------------------------------------
        return

    def evaluate(self, context: Context) -> torch.Tensor:
        """
        Compute the loss for this batch of known points.

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
        predictions = torch.concat([context[key] for key in self.input_keys], dim=1)
        targets = torch.concat([context[key] for key in self.target_keys], dim=1)

        # ---------------------------------------------------------------------------
        return self.reduction.evaluate(predictions, targets)

#####################################################################################
