#####################################################################################
import csv
import os
import torch

import pandas as pd

from typing import Dict, List, Optional, Any

from sciml.core.callback import CallbackBase
from sciml.utils import validation

#####################################################################################
class Logger(CallbackBase):
    """
    Callback that records raw training data to disk as training
    progresses, split into two separate files with different purposes:

    - history file (CSV): one row per recorded iteration, with a
      fixed header made of the loss and metric names. Only loss and
      metric values are stored here — no network or optimizer
      information. Meant to be reloaded later (e.g. via pandas) for
      analysis and plotting.
    - info file (plain text): network and optimizer information (e.g.
      number of trainable parameters, learning rate), written once,
      at the start of training. Meant purely for a human to read directly
      — it is never parsed back by the library.
    
    Examples
    --------
    >>> logger = Logger(
    ...     directory="logs/",
    ...     loss_names=["boundary_condition", "residual"],
    ...     metric_names=["relative_l2"],
    ...     frequency=10,
    ... )
    >>> # once, at the start of training:
    >>> logger.on_train_start(network=network, optimizer=optimizer)
    >>> # inside the training loop, once per iteration:
    >>> logger.on_iteration_end(
    ...     iteration=iteration,
    ...     losses={"boundary_condition": 0.012, "residual": 0.031},
    ...     metrics={"relative_l2": 0.0041},   # None if not computed this iteration
    ... )
    >>> df = logger.to_dataframe()
    """

    def __init__(
            self,
            directory: str,
            loss_names: List[str],
            metric_names: Optional[List[str]] = None,
            frequency: int = 1,
            history_filename: str = "history.csv",
            info_filename: str = "info.txt",
        ) -> None:
        """
        Parameters
        ----------
        directory : str
            Directory where both files are saved. Created automatically
            if it doesn't exist.
        loss_names : List[str]
            Names of every loss to be recorded. Fixes the corresponding
            columns of the history file.
        metric_names : List[str], optional
            Names of every validation metric to be recorded, if any.
            Fixes the corresponding columns of the history file. Rows
            recorded without metrics leave these columns empty.
        frequency : int, default=1
            Record a row in the history file every this many iterations.
            ``1`` records every iteration.
        history_filename : str, default="history.csv"
            Name of the CSV history file, inside ``directory``.
        info_filename : str, default="info.txt"
            Name of the plain-text info file, inside ``directory``.
        """
        # ---------------------------------------------------------------------------
        # > Validation
        validation.is_string(directory)
        validation.is_iterable(loss_names, dtype=str)

        if not loss_names:
            raise ValueError("loss_names must contain at least one name")
        
        if metric_names is not None:
            validation.is_iterable(metric_names, dtype=str)

        validation.is_integer(frequency)
        if frequency <= 0:
            raise ValueError("frequency must be a positive integer")
        
        validation.is_string(history_filename)
        validation.is_string(info_filename)

        # ---------------------------------------------------------------------------
        # > Inputs
        self.loss_names = list(loss_names)
        self.metric_names = list(metric_names) if metric_names is not None else []
        self.frequency = frequency
        self.history_filename = history_filename
        self.info_filename = info_filename
        self.columns = ["iteration"] + self.loss_names + self.metric_names

        self.set_directory(directory)

        return

    def set_directory(self, directory: str) -> None:
        """
        Change the base directory used for future writes, and reset the
        logger's state.

        This is useful, for example, when training an ensemble of models:
        each ensemble member can point the same ``Logger`` instance (or a
        freshly configured one) to its own subdirectory, without mixing
        histories between runs.

        Parameters
        ----------
        directory : str
            New directory where both files will be saved. Created
            automatically if it doesn't exist.
        """
        # ---------------------------------------------------------------------------
        validation.is_string(directory)

        # ---------------------------------------------------------------------------
        self.directory = directory.rstrip("/")
        os.makedirs(self.directory, exist_ok=True)
        self.history_path = f"{self.directory}/{self.history_filename}"
        self.info_path = f"{self.directory}/{self.info_filename}"

        # ---------------------------------------------------------------------------
        self.reset()

        # ---------------------------------------------------------------------------
        return

    def reset(self) -> None:
        """
        Clear the logger's state: rewrite the history file with only its
        header, and remove the info file (if it exists), starting
        completely fresh.
        """
        # ---------------------------------------------------------------------------
        with open(self.history_path, "w", newline="") as file:
            csv.writer(file).writerow(self.columns)

        if os.path.exists(self.info_path):
            os.remove(self.info_path)

        # ---------------------------------------------------------------------------
        return

    @staticmethod
    def _network_stats(network: torch.nn.Module) -> Dict[str, Any]:
        trainable = sum(p.numel() for p in network.parameters() if p.requires_grad)
        total = sum(p.numel() for p in network.parameters())
        first_param = next(network.parameters())

        return {
            "network_class": type(network).__name__,
            "num_trainable_params": trainable,
            "num_total_params": total,
            "dtype": str(first_param.dtype),
            "device": str(first_param.device),
        }

    @staticmethod
    def _optimizer_stats(optimizer: torch.optim.Optimizer) -> Dict[str, Any]:
        stats = {"optimizer_class": type(optimizer).__name__}

        groups = optimizer.param_groups
        for i, group in enumerate(groups):
            prefix = "" if len(groups) == 1 else f"group{i}_"
            for key, value in group.items():
                if key != "params":
                    stats[f"{prefix}{key}"] = value

        return stats

    def on_train_start(
            self,
            network: torch.nn.Module,
            optimizers: List[torch.optim.Optimizer],
        ) -> None:
        """
        Write network and optimizer statistics to the info file, once, at
        the start of training. Overwrites any previous info file.

        Parameters
        ----------
        network : torch.nn.Module
            Network being trained.
        optimizer : torch.optim.Optimizer
            Optimizer used to train the network.
        """
        # ---------------------------------------------------------------------------
        validation.is_network(network)
        validation.is_iterable(optimizers, dtype=torch.optim.Optimizer)

        # ---------------------------------------------------------------------------
        network_stats = self._network_stats(network)

        with open(self.info_path, "w") as file:
            file.write("====================\n")
            file.write("Training information\n")
            file.write("====================\n")

            file.write("\n")
            file.write("Neural Network\n")
            file.write("--------------\n")
            for key, value in network_stats.items():
                file.write(f"{key}: {value}\n")
            
            file.write("\n")
            file.write("Optimizer\n")
            file.write("---------\n")
            for optimizer in optimizers:
                optimizer_stats = self._optimizer_stats(optimizer)
                
                for key, value in optimizer_stats.items():
                    file.write(f"{key}: {value}\n")
                file.write("\n")

        # ---------------------------------------------------------------------------
        return

    def on_iteration_start(self, *args, **kwargs) -> None:
        """No-op: this callback only acts at the end of each iteration."""
        # ---------------------------------------------------------------------------
        return

    def on_iteration_end(
        self,
        iteration: int,
        losses: Dict[str, float],
        metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Record a row of raw loss/metric values, if the current iteration
        is due according to ``frequency``.

        Parameters
        ----------
        iteration : int
            Current iteration number.
        losses : Dict[str, float]
            Raw loss values at this iteration, keyed by loss name. Keys
            must exactly match ``self.loss_names``.
        metrics : Dict[str, float], optional
            Raw validation metric values at this iteration, keyed by
            metric name. Keys must be a subset of ``self.metric_names``.
            Pass ``None`` on iterations where validation was not
            performed.

        Raises
        ------
        ValueError
            If ``losses`` doesn't contain exactly the expected names, or
            if ``metrics`` contains a name not declared in
            ``self.metric_names``.
        """
        # ---------------------------------------------------------------------------
        if iteration % self.frequency != 0:
            return

        # ---------------------------------------------------------------------------
        validation.is_mapping(losses)
        if set(losses.keys()) != set(self.loss_names):
            raise ValueError(
                f"losses keys {sorted(losses.keys())} do not match the "
                f"declared loss_names {sorted(self.loss_names)}"
            )

        # ---------------------------------------------------------------------------
        if metrics is not None:
            validation.is_mapping(metrics)
            unknown = set(metrics.keys()) - set(self.metric_names)
            if unknown:
                raise ValueError(
                    f"metrics contain undeclared name(s) {sorted(unknown)}; "
                    f"declared metric_names are {sorted(self.metric_names)}"
                )

        # ---------------------------------------------------------------------------
        row = {"iteration": iteration, **losses, **(metrics or {})}
        ordered_row = [row.get(column, "") for column in self.columns]

        with open(self.history_path, "a", newline="") as file:
            csv.writer(file).writerow(ordered_row)

        # ---------------------------------------------------------------------------
        return

    def on_train_end(self, *args, **kwargs) -> None:
        """No-op: data is already saved progressively during training."""
        # ---------------------------------------------------------------------------
        return

    def on_exception(self, iteration: int, exception: Exception) -> None:
        """
        Record an exception that occurred during training.

        Parameters
        ----------
        iteration : int
            Iteration number at which the exception occurred.
        exception : Exception
            The exception object that was raised.
        """

        with open(self.info_path, "w") as file:
            file.write("Exception\n")
            file.write("=========\n")
            file.write(f"Iteration: {iteration}\n")
            file.write(f"Message: {exception}\n")
        
        # ---------------------------------------------------------------------------
        return

    def to_dataframe(self) -> pd.DataFrame:
        """
        Load the recorded history file into a pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            One row per recorded iteration, with the loss/metric columns
            declared at construction time.
        """
        # ---------------------------------------------------------------------------
        return pd.read_csv(self.history_path)

#####################################################################################
