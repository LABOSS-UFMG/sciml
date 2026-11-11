# -------------------------------------------------------------------------------- #
import torch

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, Optional, Sequence

from sciml.core.strategy import Strategy
from sciml.utils.checkpoint import load_checkpoint, save_checkpoint

# -------------------------------------------------------------------------------- #
class Checkpoint:
    """
    Save and restore a network and its strategy optimizers by iteration.
    """

    def __init__(
            self,
            path: str | Path,
            model: torch.nn.Module,
            strategies: Sequence[Strategy],
            prefix: str = "checkpoint",
        ) -> None:
        """
        Parameters
        ----------
        path : str or Path
            Directory where checkpoint files are stored.
        model : torch.nn.Module
            Module whose parameters and buffers are saved, e.g. ``model.network``.
        strategies : Sequence[Strategy]
            Training strategies. Each name identifies one optimizer in the file.
        prefix : str, default="checkpoint"
            Filename prefix. Each file is named ``<prefix>_<iteration>.pt``.
        """
        # ------------------------------------------------------------------------ #
        # > Set the destination directory and filename prefix
        if not prefix or Path(prefix).name != prefix:
            raise ValueError("prefix must be a non-empty filename component.")
        self.path = Path(path)
        self.prefix = prefix

        # > Identify optimizers by strategy name
        self.model = model
        self.optimizers = {strategy.name: strategy.optimizer for strategy in strategies}
        if len(self.optimizers) != len(strategies):
            raise ValueError("Checkpoint strategy names must be unique.")

    def _file_path(self, iteration: int) -> Path:
        return self.path / f"{self.prefix}_{iteration}.pt"

    def save(self, iteration: int) -> None:
        """
        Save the current state and the number of the last completed iteration.

        Parameters
        ----------
        iteration : int
            Last completed training iteration.
        """
        # ------------------------------------------------------------------------ #
        # > Replace the previous checkpoint only after serialization succeeds
        self.path.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=self.path) as directory:
            temporary_path = Path(directory) / "checkpoint.pt"
            save_checkpoint(str(temporary_path), self.model, self.optimizers,
                            metadata={"iteration": iteration})
            temporary_path.replace(self._file_path(iteration))

    def load(self, iteration: int, device: Optional[str] = None) -> Dict[str, Any]:
        """
        Restore the network and optimizers in place.

        Parameters
        ----------
        iteration : int
            Iteration whose checkpoint should be restored.
        device : str, optional
            Device to which the saved tensors are mapped.

        Returns
        -------
        Dict[str, Any]
            Saved metadata, including ``iteration`` for files written by this class.
        """
        # ------------------------------------------------------------------------ #
        return load_checkpoint(str(self._file_path(iteration)), self.model, self.optimizers, device)

# -------------------------------------------------------------------------------- #
