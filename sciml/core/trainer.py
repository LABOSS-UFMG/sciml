# -------------------------------------------------------------------------------- #
import torch

from typing import Dict, Sequence, Optional

from sciml.interfaces.model import ModelBase
from sciml.interfaces.callback import CallbackBase
from sciml.core.checkpoint import Checkpoint
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
            callbacks: Sequence[CallbackBase] = [],
            results_path: Optional[str] = None,
            checkpoint: Optional[Checkpoint] = None,
            checkpoint_interval: int = 100,
            validation: Optional[Sequence[Objective]] = None,
            validation_interval: int = 100,
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
        callbacks : Sequence[CallbackBase], default=[]
            Callbacks to be executed during training.
        results_path : Optional[str], default=None
            Path to a logs files for recording training progress. If None,
            no logging is performed.
        checkpoint : Checkpoint, optional
            Checkpoint configured with the network and strategies used for training.
        checkpoint_interval : int, default=100
            Save at multiples of this many global iterations and at the last
            completed iteration of each fit call.
        validation : Sequence[Objective], optional
            Objectives evaluated after optimization. Each uses one batch from
            its own sampler. Use a fixed full batch for comparable measurements.
        validation_interval : int, default=100
            Evaluate at multiples of this many global iterations and at the
            last completed iteration of each fit call.
        """
        # ------------------------------------------------------------------------ #
        # Validation
        if not isinstance(checkpoint_interval, int) or checkpoint_interval <= 0:
            raise ValueError("checkpoint_interval must be a positive integer.")
        if type(validation_interval) is not int or validation_interval <= 0:
            raise ValueError("validation_interval must be a positive integer.")
        if validation and results_path and any(s.name == "validation" for s in strategies):
            raise ValueError("The strategy name 'validation' conflicts with validation_log.csv.")
        
        # ------------------------------------------------------------------------ #
        # Store constructor arguments
        self.model = model
        self.objectives = objectives
        self.strategies = strategies
        self.callbacks = callbacks
        self.results_path = results_path
        self.checkpoint = checkpoint
        self.checkpoint_interval = checkpoint_interval
        self.validation = validation if validation is not None else ()
        self.validation_interval = validation_interval

        # ------------------------------------------------------------------------ #
        # Internal parameters
        self.iteration = 0

        # ------------------------------------------------------------------------ #
        return

    def _validate(self) -> Dict[str, float]:
        """Evaluate held-out losses with input derivatives enabled."""
        network = self.model if isinstance(self.model, torch.nn.Module) else getattr(self.model, "network", None)
        modes = {}
        losses = {}
        try:
            if isinstance(network, torch.nn.Module):
                modes = {module: module.training for module in network.modules()}
                network.eval()

            with torch.enable_grad():
                for objective in self.validation:
                    evaluation = objective.evaluate(self.model)
                    for name, value in evaluation.losses.items():
                        key = f"{evaluation.name}.{name}"
                        if key in losses:
                            raise ValueError(f"Duplicate validation loss column: {key}")
                        losses[key] = value
        finally:
            for module, training in modes.items():
                module.training = training
        return losses

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
        # Callbacks
        for callback in self.callbacks:
            callback.on_train_start()

        # ------------------------------------------------------------------------ #
        try:

            for iteration in range(num_iterations):

                # ---------------------------------------------------------------- #
                # Callbacks
                for callback in self.callbacks:
                    callback.on_iteration_start()

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

                # ---------------------------------------------------------------- #
                # Callbacks
                for callback in self.callbacks:
                    callback.on_iteration_end()

                if self.results_path:
                    progress.log(self.results_path, self.iteration, steps)

                # ---------------------------------------------------------------- #
                # Evaluate validation objectives once after all optimization steps
                if self.validation and (
                        self.iteration % self.validation_interval == 0
                        or iteration == num_iterations - 1
                    ):
                    losses = self._validate()
                    if self.results_path:
                        progress.log_validation(self.results_path, self.iteration, losses)

                # ---------------------------------------------------------------- #
                # Save periodic and final checkpoints after all strategies finish
                if self.checkpoint is not None and (
                        self.iteration % self.checkpoint_interval == 0
                        or iteration == num_iterations - 1
                    ):
                    self.checkpoint.save(self.iteration)
        
        # ------------------------------------------------------------------------ #
        except Exception:

            # ---------------------------------------------------------------- #
            # Callbacks
            for callback in self.callbacks:
                callback.on_exception()

            raise

        # ------------------------------------------------------------------------ #
        # Callbacks
        for callback in self.callbacks:
            callback.on_train_end()

        # ------------------------------------------------------------------------ #
        return

# -------------------------------------------------------------------------------- #
