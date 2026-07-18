#####################################################################################
import csv
import os

import pandas as pd

from typing import Optional, Iterable

from sciml.core.callback import CallbackBase
from sciml.core.evaluation import Evaluation
from sciml.utils import checker

#####################################################################################
class Logger(CallbackBase):
    """
    Callback that records raw training data to disk as training
    progresses, split into two separate files with different purposes:

    Examples
    --------
    >>> logger = Logger(
    ...     directory="logs/",
    ... )
    >>> df = logger.to_dataframe()
    """

    def __init__(
            self,
            directory: str,
        ) -> None:
        """
        Parameters
        ----------
        directory : str
            Directory where both files are saved. Created automatically
            if it doesn't exist.
        """
        # ---------------------------------------------------------------------------
        # > Validation
        checker.is_string(directory)

        # ---------------------------------------------------------------------------
        # > Inputs
        self.directory = directory

        # ---------------------------------------------------------------------------
        # > Create directory
        self.directory = directory.rstrip("/")
        os.makedirs(self.directory, exist_ok=True)

        # ---------------------------------------------------------------------------
        # > Internal parameters
        self._losses_file = False
        self._metrics_file = False

        # ---------------------------------------------------------------------------
        return
    
    def on_train_start(self, *args, **kwargs) -> None:
        # ---------------------------------------------------------------------------
        return

    def on_iteration_start(self, *args, **kwargs) -> None:
        # ---------------------------------------------------------------------------
        return

    def on_iteration_end(
            self,
            iteration: int,
            losses: Iterable[Evaluation],
            metrics: Optional[Iterable[Evaluation]] = None,
        ) -> None:
        """
        Record a row of raw loss/metric values, if the current iteration
        is due according to ``frequency``.

        Parameters
        ----------
        iteration : int
            Current iteration number.
        losses : Iterable[Evaluation]
            Each evaluation represents an objetive function.
        metrics : Iterable[Evaluation], optional
            Each evaluation represents a validation dataset.
        """
        # ---------------------------------------------------------------------------
        checker.is_integer(iteration)
        checker.is_iterable(losses, dtype=Evaluation)

        if metrics is not None:
            checker.is_iterable(metrics, dtype=Evaluation)
        
        # ---------------------------------------------------------------------------
        if not self._losses_file:
            create_file = True
            self._losses_file = True
        else:
            create_file = False
            
        for loss in losses:

            filename = self.directory + "/" + loss.metadata["name"] + ".csv"

            if create_file:
                header = ["iteration", "Objective function"] + list(loss.values.keys())
                with open(filename, "w", newline="") as file:
                    csv.writer(file).writerow(header)
            
            row = [
                iteration,
                sum([loss.values[key] * loss.weights[key] for key in loss.values.keys()]),
            ] + [v for v in loss.values.values()]
            with open(filename, "a", newline="") as file:
                csv.writer(file).writerow(row)
                
        # Only iterate over metrics if it's not None
        if metrics is not None:

            if not self._metrics_file:
                create_file = True
                self._metrics_file = True
            else:
                create_file = False

            for metric in metrics:

                filename = self.directory + "/" + metric.metadata["name"] + ".csv"

                if create_file:
                    header = ["iteration"] + list(metric.values.keys())
                    with open(filename, "w", newline="") as file:
                        csv.writer(file).writerow(header)
                
                if None in metric.values.values():
                    continue
                
                row = [iteration] + [v for v in metric.values.values()]
                with open(filename, "a", newline="") as file:
                    csv.writer(file).writerow(row)

        # ---------------------------------------------------------------------------
        return

    def on_train_end(self, *args, **kwargs) -> None:
        # ---------------------------------------------------------------------------
        return

    def on_exception(self, *args, **kwargs) -> None:
        # ---------------------------------------------------------------------------
        return

    def to_dataframe(self, name: str) -> pd.DataFrame:
        """
        Load the recorded history file into a pandas DataFrame.

        Parameters
        ----------
        name : str
            Name of the file to load, without the ``.csv`` extension.
        
        Returns
        -------
        pd.DataFrame
            One row per recorded iteration, with the loss/metric columns
            declared at construction time.
        """
        # ---------------------------------------------------------------------------
        checker.is_string(name)

        # ---------------------------------------------------------------------------
        return pd.read_csv(self.directory + "/" + name + ".csv")

#####################################################################################
