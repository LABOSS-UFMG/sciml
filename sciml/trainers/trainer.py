#####################################################################################
from typing import Iterable, Optional

from sciml.core.trainer import TrainerBase
from sciml.core.model import ModelBase
from sciml.core.objective import Objective
from sciml.core.strategy import StrategyBase
from sciml.core.callback import CallbackBase
from sciml.core.validation import Validation
from sciml.core.evaluation import Evaluation

from sciml.callbacks.logger import Logger
from sciml.callbacks.checkpoint import Checkpoint

from sciml.utils import checker

#####################################################################################
class Trainer(TrainerBase):
    """
    Trainer class that handles the training process of a model using specified
    objectives, strategies, validations, logger, checkpoint, and callbacks.
    """

    def __init__(
            self,
            model: ModelBase,
            objectives: Iterable[Objective],
            strategies: Iterable[StrategyBase],
            validations: Optional[Iterable[Validation]] = None,
            logger: Optional[Logger] = None,
            checkpoint: Optional[Checkpoint] = None,
            callbacks: Optional[Iterable[CallbackBase]] = None
        ) -> None:
        """
        Parameters
        ----------
        model : ModelBase
            Model to be trained.
        objectives : Iterable[Objective]
            Objectives to be optimized.
        strategies : Iterable[StrategyBase]
            Strategies to be used for optimization.
        validations : Optional[Iterable[Validation]], optional
            Validations to be used for evaluation, by default None
        logger : Optional[Logger], optional
            Logger to be used for logging, by default None
        checkpoint : Optional[Checkpoint], optional
            Checkpoint to be used for saving, by default None
        callbacks : Iterable[CallbackBase], optional
            Callbacks to be used for additional functionality, by default []
        """
        # ---------------------------------------------------------------------------
        # > Validation
        checker.is_dtype(model, ModelBase)
        checker.is_iterable(objectives, dtype=Objective)
        checker.is_iterable(strategies, dtype=StrategyBase)

        if not (logger is None):
            checker.is_dtype(logger, Logger)
        
        if not (checkpoint is None):
            checker.is_dtype(checkpoint, Checkpoint)
        
        if not (callbacks is None):
            checker.is_iterable(callbacks, dtype=CallbackBase)        
        
        # ---------------------------------------------------------------------------
        # > Inputs
        self.model = model
        self.objectives = objectives
        self.strategies = strategies
        self.validations = validations
        self.logger = logger
        self.checkpoint = checkpoint
        self.callbacks = callbacks

        # ---------------------------------------------------------------------------
        # > Internal parameters
        self._iteration = 0

        # ---------------------------------------------------------------------------
        return

    def fit(
            self,
            num_iterations: int,
            verbose: bool = False,
            verbose_frequency: int = 100,
        ) -> None:
        """
        Train the model for a specified number of iterations, applying strategies,
        objectives, validations, and callbacks as defined in the Trainer.

        Parameters
        ----------
        num_iterations : int
            Number of iterations to train the model.
        verbose : bool, optional
            If True, prints progress information during training, by default False
        """
        # ---------------------------------------------------------------------------
        # > Validate
        checker.is_integer(num_iterations)
        checker.is_boolean(verbose)
        checker.is_integer(verbose_frequency)

        # ---------------------------------------------------------------------------
        # > Callbacks
        if not (self.callbacks is None):
            for callback in self.callbacks:
                callback.on_train_start()
        
        # ---------------------------------------------------------------------------
        try:

            for _ in range(num_iterations):
                self._iteration += 1

                # ---------------------------------------------------------------------------
                # > Callbacks
                if not (self.callbacks is None):
                    for callback in self.callbacks:
                        callback.on_iteration_start()
        
                # -------------------------------------------------------------------
                # > Optimize parameters
                losses = []
                for strategy in self.strategies:
                    evaluation = strategy.step(self.model, self.objectives)
                    if not (evaluation is None):
                        losses.append(evaluation)

                # -------------------------------------------------------------------
                # > Validation
                metrics = None
                if not (self.validations is None):
                    metrics: Iterable[Evaluation] = []
                    for validation in self.validations:
                        evaluation = validation.evaluate(self._iteration, self.model)
                        metrics.append(evaluation)
                
                # -------------------------------------------------------------------
                # > Callbacks
                if self.logger is not None:
                    self.logger.on_iteration_end(self._iteration, losses, metrics)
                
                if self.checkpoint is not None:
                    self.checkpoint.on_iteration_end(self._iteration)

                if not (self.callbacks is None):
                    for callback in self.callbacks:
                        callback.on_iteration_end()
                
                # -------------------------------------------------------------------
                # > Show progress
                if (verbose and (self._iteration % verbose_frequency == 0)) or (self._iteration == 1):
                    width = 14
                    
                    # Formats
                    def format_header(text: str) -> str:
                        text = str(text)
                        if len(text) > width:
                            text = text[:width - 1] + "."
                        return f"{text:>{width}}"

                    def format_value(value) -> str:
                        if value is None:
                            return f"{'-':>{width}}"
                        if isinstance(value, int):
                            return f"{value:>{width}}"
                        else:
                            return f"{float(value):>{width}.6e}"

                    # Process evaluation
                    def process_loss(e: Evaluation):
                        keys = list(e.values.keys())
                        name = e.metadata["name"]
                        total = sum([e.values[k] * e.weights[k] for k in keys])
                        values = [v for v in e.values.values()]
                        return (total, name, values, keys)

                    # Losses
                    if len(losses) == 1:
                        total, name, values, keys = process_loss(losses[0])
                        header = ["Iteration", name] + keys
                        itens = [self._iteration, total] + values
                    else:
                        header = ["Iteration"]
                        itens = [self._iteration]
                        for loss in losses:
                            total, name, values, keys = process_loss(loss)
                            header.append(name)
                            itens.append(total)
                    
                    # Metrics
                    if not (metrics is None):
                        for metric in metrics:
                            header.extend(list(metric.values.keys()))
                            itens.extend(list(metric.values.values()))
                    
                    # Show progress
                    if self._iteration == 1:
                        num_titles = len(header)
                        header = " ".join(format_header(name) for name in header)
                        separator = " ".join("-" * width for _ in range(num_titles))
                        print(header)
                        print(separator)
                    
                    row = " ".join(format_value(item) for item in itens)
                    print(row)
        
        # ---------------------------------------------------------------------------
        except Exception as exception:

            # -----------------------------------------------------------------------
            # > Callbacks
            if self.checkpoint is not None:
                self.checkpoint.on_exception(self._iteration)

            if not (self.callbacks is None):
                for callback in self.callbacks:
                    callback.on_exception()

            # -----------------------------------------------------------------------
            # > Exception
            raise
        
        # ---------------------------------------------------------------------------
        # > Callbacks
        if self.checkpoint is not None:
            self.checkpoint.on_train_end(self._iteration)
        
        if not (self.callbacks is None):
            for callback in self.callbacks:
                callback.on_train_end()
        
        # ---------------------------------------------------------------------------
        return

#####################################################################################
