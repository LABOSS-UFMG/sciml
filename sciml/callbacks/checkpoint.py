#####################################################################################
import torch

from typing import Union, Iterable, Dict, Optional

from sciml.core.callback import CallbackBase
from sciml.utils import serialization
from sciml.utils import checker

#####################################################################################
class Checkpoint(CallbackBase):
    """
    Callback that periodically saves the network and optimizer to disk,
    using ``sciml.utils.serialization.save_checkpoint``.

    A checkpoint is saved every ``frequency`` iterations, each to its own
    file, named after the iteration it was saved at (e.g.
    ``checkpoint_100.pt``, ``checkpoint_200.pt``, ...). No file is ever
    overwritten, so every saved checkpoint is kept.

    Examples
    --------
    Save a checkpoint every 100 iterations:

    >>> checkpoint = Checkpoint(directory="checkpoints/", frequency=100)
    >>> # inside the training loop, once per iteration:
    >>> checkpoint.on_iteration_end(
    ...     network, optimizer, iteration,
    ... )
    """

    def __init__(
            self,
            directory: str,
            frequency: int,
            network: Union[torch.nn.Module, Iterable[torch.nn.Module], Dict[str, torch.nn.Module]],
            optimizer: Optional[Union[torch.optim.Optimizer, Iterable[torch.optim.Optimizer], Dict[str, torch.optim.Optimizer]]] = None,
            filename_template: str = "checkpoint_{iteration}.pt",
        ) -> None:
        """
        Parameters
        ----------
        directory : str
            Directory where checkpoint files are saved. Created
            automatically if it doesn't exist.
        frequency : int
            Save a checkpoint every this many iterations.
        filename_template : str, default="checkpoint_{iteration}.pt"
            Filename pattern used for each saved checkpoint. Must contain
            the ``{iteration}`` placeholder, which is replaced by the
            current iteration number.
        """
        # ---------------------------------------------------------------------------
        # > Validation
        checker.is_string(directory)
        checker.is_integer(frequency)

        if frequency <= 0:
            raise ValueError("frequency must be a positive integer")
        
        checker.is_string(filename_template)
        if "{iteration}" not in filename_template:
            raise ValueError("filename_template must contain the '{iteration}' placeholder")

        # ---------------------------------------------------------------------------
        # > Inputs
        self.directory = directory.rstrip("/")
        self.frequency = frequency
        self.network = network
        self.optimizer = optimizer
        self.filename_template = filename_template

        # ---------------------------------------------------------------------------
        return

    def _path_for(self, iteration: int) -> str:
        """
        Build the destination path for a checkpoint saved at ``iteration``.

        Parameters
        ----------
        iteration : int
            Iteration number to embed in the filename.

        Returns
        -------
        str
            Full path where the checkpoint should be saved.
        """
        # ---------------------------------------------------------------------------
        filename = self.filename_template.format(iteration=iteration)

        # ---------------------------------------------------------------------------
        return f"{self.directory}/{filename}"

    def on_train_start(self, *args, **kwargs) -> None:
        """No-op: this callback only acts at the end of each iteration."""
        # ---------------------------------------------------------------------------
        return

    def on_iteration_start(self, *args, **kwargs) -> None:
        """No-op: this callback only acts at the end of each iteration."""
        # ---------------------------------------------------------------------------
        return

    def on_iteration_end(
            self,
            iteration: int,
        ) -> None:
        """
        Save the network and optimizer if the current iteration is a
        multiple of ``frequency``.

        Parameters:
        -----------
        network : torch.nn.Module
            Current neural network
        optimizer : torch.optim.Optimizer
            Current optimizer
        iteration : int
            Current iteration
        """
        # ---------------------------------------------------------------------------
        if iteration % self.frequency == 0:
            serialization.save_checkpoint(
                self._path_for(iteration),
                network=self.network,
                optimizer=self.optimizer,
            )

        # ---------------------------------------------------------------------------
        return

    def on_train_end(
            self,
            iteration: int,
        ) -> None:
        """
        Save the network and optimizer on training end.

        Parameters:
        -----------
        network : torch.nn.Module
            Current neural network
        optimizer : torch.optim.Optimizer
            Current optimizer
        iteration : int
            Current iteration
        """
        # ---------------------------------------------------------------------------
        if iteration % self.frequency != 0:
            serialization.save_checkpoint(
                self._path_for(iteration),
                network=self.network,
                optimizer=self.optimizer,
            )

        # ---------------------------------------------------------------------------
        return

    def on_exception(
            self,
            iteration: int,
        ) -> None:
        """
        Best-effort save of a recovery checkpoint when training fails.

        Parameters:
        -----------
        network : torch.nn.Module
            Current neural network
        optimizer : torch.optim.Optimizer
            Current optimizer
        iteration : int
            Current iteration
        """
        # ---------------------------------------------------------------------------
        serialization.save_checkpoint(
            f"{self.directory}/recovery_{iteration}.pt",
            network=self.network,
            optimizer=self.optimizer,
        )

        # ---------------------------------------------------------------------------
        return

#####################################################################################
