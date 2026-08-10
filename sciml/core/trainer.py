# -------------------------------------------------------------------------------- #
from typing import Sequence, Optional

from sciml.interfaces.model import ModelBase
from sciml.core.objective import Objective
from sciml.core.strategy import Strategy
from sciml.core.progress import Progress

# -------------------------------------------------------------------------------- #
class Trainer():
    """
    Execute the training of a model using specified objectives and strategies.
    """

    def __init__(
            self,
            model: ModelBase,
            objectives: Sequence[Objective],
            strategies: Sequence[Strategy],
            results_path: Optional[str] = None,
        ) -> None:
        """
        Parameters
        ----------
        model : ModelBase
            Model to be trained.
        objectives : Sequence[Objective]
            Objectives to be optimized.
        strategies : Sequence[StrategyBase]
            Strategies to be used for optimization.
        results_path : Optional[str], default=None
            Path to a logs files for recording training progress. If None,
            no logging is performed.
        """
        
        # ------------------------------------------------------------------------ #
        # Store constructor arguments
        self.model = model
        self.objectives = objectives
        self.strategies = strategies
        self.results_path = results_path

        # ------------------------------------------------------------------------ #
        # Internal parameters
        self.iteration = 0

        # ------------------------------------------------------------------------ #
        return

    def fit(
            self,
            num_iterations: int,
            verbose: bool = False,
            interval: int = 100,
        ) -> None:
        """
        Train the model for a specified number of iterations, applying strategies,
        objectives, validations, and callbacks as defined in the Trainer.

        Parameters
        ----------
        num_iterations : int
            Number of iterations to train the model.
        verbose : bool, default=False
            If True, prints training progress to the console.
        """
        # ------------------------------------------------------------------------ #
        # Initialize progress tracker
        progress = Progress(self.strategies)

        # ------------------------------------------------------------------------ #
        try:

            for iteration in range(num_iterations):
                # ---------------------------------------------------------------- #
                # Increase iteration
                self.iteration += 1

                # ---------------------------------------------------------------- #
                # Optimize parameters
                steps = []
                for strategy in self.strategies:
                    step = strategy.step(self.iteration, self.model, self.objectives)
                    if step: steps.append(step)

                # ---------------------------------------------------------------- #
                # Update progress tracker and show progress
                progress.update(steps)

                if verbose:
                    if (
                        (self.iteration == 1) or
                        (self.iteration % interval == 0) or
                        (iteration == num_iterations - 1)
                        ):
                        progress.display(self.iteration)

                if self.results_path:
                    progress.log(self.results_path, self.iteration, steps)
        
        # ------------------------------------------------------------------------ #
        except Exception:
            raise

        # ------------------------------------------------------------------------ #
        return

# -------------------------------------------------------------------------------- #
