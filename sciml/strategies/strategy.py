#####################################################################################
import torch

from typing import Iterable, Optional, Callable

from sciml.core.strategy import StrategyBase
from sciml.core.model import ModelBase
from sciml.core.objective import Objective
from sciml.core.evaluation import Evaluation
from sciml.utils import checker

#####################################################################################
class Strategy(StrategyBase):
    """
    Optimization strategy used during training.

    A ``Strategy`` defines how one or more objectives are evaluated and how
    their losses are combined to update the model parameters.
    """

    def __init__(
            self,
            optimizer: torch.optim.Optimizer,
            keys: Optional[Iterable[str]] = None,
            name: Optional[str] = None,
            enable: Optional[Callable[[int], bool]] = None
        ) -> None:
        """
        Parameters
        ----------
        optimizer : torch.optim.Optimizer
            Optimizer responsible for updating the model parameters.
        keys : Iterable[str] or None, default=None
            Names of the losses included in the objective function. If
            ``None``, all losses associated with each objective are
            considered.
        enable : Callable[[int, Evaluation], bool] or None, default=None
            Function that determines whether the strategy is enabled at a
            given training step. It takes the current step. If ``None``,
            the strategy is always enabled.
        """
        # -------------------------------------------------------------------------
        # > Validation
        checker.is_dtype(optimizer, torch.optim.Optimizer)

        if not (name is None):
            checker.is_string(name)

        if keys is not None:
            checker.is_iterable(keys, dtype=str)

        # -------------------------------------------------------------------------
        # > Inputs
        self.optimizer = optimizer
        self.keys = keys
        self.name = name
        self.enable = enable

        # -------------------------------------------------------------------------
        # > Internal parameters
        self._evaluations: Iterable[Evaluation] = []
        self._current_step: int = 0

        # -------------------------------------------------------------------------
        return

    # Auxiliary method
    def evaluate(
            self,
            model: ModelBase,
            objectives: Iterable[Objective],
        ) -> Iterable[Evaluation]:
        """
        Evaluate all objectives required by this strategy.

        Parameters
        ----------
        model : ModelBase
            Model to be evaluated.
        objectives : Iterable[Objective]
            Collection of objectives evaluated by the strategy.

        Returns
        -------
        dict[str, Context]
            Dictionary mapping each objective name to its corresponding
            evaluation context.
        """

        evaluations = []
        for objective in objectives:
            evaluation = objective.evaluate(model, self.keys)

            if evaluation is not None:
                evaluations.append(evaluation)

        return evaluations

    # Auxiliary method
    def objective_function(
            self,
            evaluations: Iterable[Evaluation],
        ) -> torch.Tensor:
        """
        Compute the weighted objective function.

        Parameters
        ----------
        contexts : dict[str, Context]
            Evaluation contexts produced by the objectives.

        Returns
        -------
        torch.Tensor
            Scalar objective function obtained by combining the weighted
            losses from all evaluation contexts.
        """

        obj_function = 0.0
        for evaluation in evaluations:
            obj_function += sum(
                evaluation.values[key] * evaluation.weights[key]
                for key in evaluation.values.keys()
            )

        return obj_function

    # Auxiliary method
    def closure(
            self,
            model: ModelBase,
            objectives: Iterable[Objective],
        ) -> torch.Tensor:
        """
        Evaluate the objective function and compute its gradients.

        This method is intended to be passed to PyTorch optimizers that
        require a closure (e.g. ``LBFGS``). It evaluates the objectives,
        computes the objective function and performs backpropagation.

        Parameters
        ----------
        model : ModelBase
            Model to be optimized.
        objectives : Iterable[Objective]
            Collection of objectives evaluated by the strategy.

        Returns
        -------
        torch.Tensor
            Scalar objective function evaluated at the current model
            parameters.
        """

        self.optimizer.zero_grad()
        self._evaluations = self.evaluate(model, objectives)
        obj_function = self.objective_function(self._evaluations)
        obj_function.backward()

        return obj_function

    # Required method
    def step(
            self,
            model: ModelBase,
            objectives: Iterable[Objective],
        ) -> Optional[Evaluation]:
        """
        Perform a single optimization step.

        Parameters
        ----------
        model : ModelBase
            Model to be optimized.
        objectives : Iterable[Objective]
            Collection of objectives evaluated during the optimization
            step.

        Returns
        -------
        dict[str, Context] or None
            Dictionary containing the evaluation context associated with
            each objective.
        """
        self._current_step += 1

        if (self.enable is None) or self.enable(self._current_step):

            self.optimizer.step(
                lambda: self.closure(model, objectives)
            )

            evaluation = Evaluation(
                values={k: v.item() for e in self._evaluations for k, v in e.values.items()},
                weights={k: v for e in self._evaluations for k, v in e.weights.items()},
                metadata={"name": self.name if not (self.name is None) else "training"},
            )

            return evaluation

        else:
            return None

#####################################################################################
