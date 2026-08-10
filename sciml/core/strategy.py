# -------------------------------------------------------------------------------- #
import torch

from typing import Sequence, Optional, Callable

from sciml.interfaces.model import ModelBase
from sciml.core.objective import Objective, Evaluation
from sciml.contracts.dataclasses import Step

# -------------------------------------------------------------------------------- #
class Strategy():
    """
    Optimization strategy used during training.
    """

    def __init__(
            self,
            name: str,
            optimizer: torch.optim.Optimizer,
            objective_names: Optional[Sequence[str]] = None,
            enable: Optional[Callable[[int], bool]] = None
        ) -> None:
        """
        Parameters
        ----------
        name : str
            Strategy name.
        optimizer : torch.optim.Optimizer
            Optimizer responsible for updating the model parameters.
        objective_names : Sequence[str] or None, default=None
            Names of the objectives included. If ``None``, all objectives are
            considered.
        enable : Callable[[int], bool] or None, default=None
            Function that determines whether the strategy is enabled at a
            given training step. It takes the current step. If ``None``,
            the strategy is always enabled.
        """
        # ------------------------------------------------------------------------ #
        # Store constructor arguments
        self.name = name
        self.optimizer = optimizer
        self.objective_names = objective_names
        self.enable = enable

        # ------------------------------------------------------------------------ #
        # Internal parameters
        self.evaluations: Sequence[Evaluation] = []
        self.current_step: int = 0

        # ------------------------------------------------------------------------ #
        return

    def step(
            self,
            iteration: int,
            model: ModelBase,
            objectives: Sequence[Objective],
        ) -> Optional[Step]:
        """
        Perform a single optimization step.

        Parameters
        ----------
        iteration : int
            Current training iteration.
        model : ModelBase
            Model to be optimized.
        objectives : Sequence[Objective]
            Collection of objectives evaluated during the optimization step.

        Returns
        -------
        Step or None
            Step parameters.
        """
        # ------------------------------------------------------------------------ #
        # Auxiliary closure function
        def closure(model: ModelBase, objectives: Sequence[Objective]):
            
            # Reset gradients
            self.optimizer.zero_grad()

            # Evaluate all objectives required by this strategy
            self.evaluations = []
            for objective in objectives:
                if (self.objective_names is None) or (objective.name in self.objective_names):
                    evaluation = objective.evaluate(model)
                    self.evaluations.append(evaluation)

            # Create objective function
            function = sum(evaluation.objective for evaluation in self.evaluations)

            # Compute gradients
            function.backward()

            return function
        
        # ------------------------------------------------------------------------ #
        # Optimize if enabled
        if (self.enable is None) or self.enable(iteration):

            # Apply optimizer
            self.optimizer.step(lambda: closure(model, objectives))

            # Define output
            step = Step(name=self.name, evaluations=self.evaluations)

            return step

        # ------------------------------------------------------------------------ #
        return None

# -------------------------------------------------------------------------------- #
