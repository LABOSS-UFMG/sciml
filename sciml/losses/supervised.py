#####################################################################################
import torch

from typing import Optional

from sciml.core.loss import LossBase
from sciml.core.contracts import Batch
from sciml.core.metric import MetricBase
from sciml.metrics import MeanSquaredError
from sciml.utils import validation

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
    ...     input_key="xt0",
    ...     target_key="u0",
    ... )
    >>> initial_loss.evaluate(network, batch)
    """

    def __init__(
            self,
            name: str,
            weight: float,
            input_key: str,
            target_key: str,
            reduction: MetricBase = MeanSquaredError(),
        ) -> None:
        """
        Parameters
        ----------
        name : str
            Short identifier used when logging or plotting this loss
            (e.g. ``"initial_condition"``, ``"boundary_condition"``,
            ``"observational"``).
        weight : float
            Scalar weight applied to this loss when combined with others.
        input_key : str
            Key used to retrieve the sampled inputs from ``batch.inputs``
            (e.g. ``"xt0"`` for initial-condition points).
        target_key : str
            Key used to retrieve the reference values from
            ``batch.targets`` (e.g. ``"u0"`` for the known initial values).
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
        validation.is_string(name)
        validation.is_float(weight)
        validation.is_string(input_key)
        validation.is_string(target_key)
        validation.is_metric(reduction)

        # ---------------------------------------------------------------------------
        # > Inputs
        self.name = name
        self.weight = weight
        self.input_key = input_key
        self.target_key = target_key
        self.reduction = reduction

        # ---------------------------------------------------------------------------
        return

    def evaluate(self, network: torch.nn.Module, batch: Batch) -> torch.Tensor:
        """
        Compute the loss for this batch of known points.

        Parameters
        ----------
        network : torch.nn.Module
            Neural network being trained.
        batch : Batch
            Must contain ``self.input_key`` in ``batch.inputs`` and
            ``self.target_key`` in ``batch.targets``.

        Returns
        -------
        torch.Tensor
            Scalar tensor representing the loss.

        Raises
        ------
        ValueError
            If ``batch.targets`` is ``None`` (this loss always requires
            known reference values).
        """
        # ---------------------------------------------------------------------------
        x = batch.inputs[self.input_key]
        target = batch.targets[self.target_key]

        prediction = network(x)

        # ---------------------------------------------------------------------------
        return self.reduction.evaluate(prediction, target)

#####################################################################################
