# -------------------------------------------------------------------------------- #
import torch

from time import perf_counter
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
            progress: Optional[Progress] = None,
            callbacks: Sequence[CallbackBase] = [],
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
        progress : Progress, optional
            Tracker for these strategies. Pending history is saved at the end of each fit call.
        callbacks : Sequence[CallbackBase], default=[]
            Callbacks to be executed during training.
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
        
        # ------------------------------------------------------------------------ #
        # Store constructor arguments
        self.model = model
        self.objectives = objectives
        self.strategies = strategies
        self.progress = progress
        self.callbacks = callbacks
        self.checkpoint = checkpoint
        self.checkpoint_interval = checkpoint_interval
        self.validation = validation if validation is not None else ()
        self.validation_interval = validation_interval

        # ------------------------------------------------------------------------ #
        # Internal parameters
        self.iteration = 0
        self.optimization_s = self.elapsed_s = 0.0

        # ------------------------------------------------------------------------ #
        return

    def fit(
            self,
            num_iterations: int,
        ) -> None:
        """
        Train the model for a specified number of iterations, applying strategies,
        objectives, validations, and callbacks as defined in the Trainer.

        Parameters
        ----------
        num_iterations : int
            Number of iterations to train the model.
        """
        # ------------------------------------------------------------------------ #
        # Callbacks
        fit_start = perf_counter()
        previous_elapsed = self.elapsed_s
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
                step_start = perf_counter()
                for strategy in self.strategies:
                    step = strategy.step(self.iteration, self.model, self.objectives)
                    if step: steps.append(step)
                self.optimization_s += perf_counter() - step_start

                # ---------------------------------------------------------------- #
                # Update progress tracker and show progress
                if self.progress is not None:
                    self.progress.update(steps, self.iteration)
                    self.progress.display_if_due(self.iteration, final=iteration == num_iterations - 1)

                # ---------------------------------------------------------------- #
                # Callbacks
                for callback in self.callbacks:
                    callback.on_iteration_end()

                # ---------------------------------------------------------------- #
                # Evaluate validation objectives once after all optimization steps
                if (
                        self.progress is not None and
                        self.validation and
                        (
                            self.iteration % self.validation_interval == 0
                            or iteration == num_iterations - 1
                        )
                    ):
                    # Compute validation losses
                    losses = {}
                    with self.model.evaluation_mode(), torch.enable_grad():
                        for objective in self.validation:
                            evaluation = objective.evaluate(self.model)
                            for name, value in evaluation.losses.items():
                                key = f"{evaluation.name}.{name}"
                                losses[key] = value

                    # Update progress
                    self.progress.update_validation(self.iteration, losses)

                # ---------------------------------------------------------------- #
                # Save periodic and final checkpoints after all strategies finish
                if self.checkpoint is not None and (
                        self.iteration % self.checkpoint_interval == 0
                        or iteration == num_iterations - 1
                    ):
                    self.checkpoint.save(self.iteration)

                self.elapsed_s = previous_elapsed + (perf_counter() - fit_start)
                if self.progress is not None:
                    self.progress.record_times(self.iteration, self.optimization_s, self.elapsed_s)

            if self.progress is not None:
                self.progress.save()
        
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
