# -------------------------------------------------------------------------------- #
from typing import Sequence

from sciml.interfaces.sampler import SamplerBase
from sciml.interfaces.loss import LossBase
from sciml.interfaces.model import ModelBase
from sciml.contracts.dataclasses import Evaluation

# -------------------------------------------------------------------------------- #
class Objective():
    """
    Represents an objective function to be optimized.
    """

    def __init__(
            self,
            name: str,
            sampler: SamplerBase,
            losses: Sequence[LossBase],
        ) -> None:
        """
        Parameters
        ----------
        name : str
            Name of the objective.
        sampler : SamplerBase
            Sampler used to generate the evaluation context.
        losses : Sequence[LossBase]
            Collection of losses used to compute the objective function.
        """
        # ------------------------------------------------------------------------ #
        # Store constructor arguments
        self.name = name
        self.sampler = sampler
        self.losses = losses

        # ------------------------------------------------------------------------ #
        return

    def evaluate(self, model: ModelBase) -> Evaluation:
        """
        Evaluate the objective function and compute its gradients.

        Parameters
        ----------
        model : ModelBase
            Model to be evaluated.

        Returns
        -------
        Evaluation
            Evaluation object containing the objective function value and
            the individual losses and their weights.
        """
        # ------------------------------------------------------------------------ #
        # Create current sample
        context = self.sampler.next()

        # Evaluate model
        model.compute(context)

        # Evaluate losses
        evaluation = Evaluation(name=self.name)

        for loss in self.losses:
            v = loss.evaluate(context)

            evaluation.objective += v * loss.weight
            evaluation.losses[loss.name] = v.item()
            evaluation.weights[loss.name] = loss.weight

        # ------------------------------------------------------------------------ #
        return evaluation

# -------------------------------------------------------------------------------- #
