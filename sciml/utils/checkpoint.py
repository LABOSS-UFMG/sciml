# -------------------------------------------------------------------------------- #
import torch

from typing import Any, Dict, Optional

# -------------------------------------------------------------------------------- #
def save_checkpoint(
        path: str,
        model: torch.nn.Module,
        optimizers: Dict[str, torch.optim.Optimizer],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
    """
    Save a model and the optimizer of every strategy to a single file.

    Parameters
    ----------
    path : str
        Path to the file where the checkpoint is written.
    model : torch.nn.Module
        Model whose parameters are saved (e.g. ``model.network``).
    optimizers : Dict[str, torch.optim.Optimizer]
        Optimizers of every strategy to save. The keys are the strategy names
        (e.g. ``strategy.name``) and the values are the corresponding optimizer
        objects (e.g. ``strategy.optimizer``).
    metadata : Dict[str, Any], optional
        Arbitrary extra information to store alongside the checkpoint
        (e.g. iteration count, random seed, problem parameters).
    """
    # ------------------------------------------------------------------------ #
    checkpoint = {
        "model": model.state_dict(),
        "optimizers": {
            key: value.state_dict() for key, value in optimizers.items()
        },
        "metadata": metadata if (metadata is not None) else {},
    }

    torch.save(checkpoint, path)

    # ------------------------------------------------------------------------ #
    return

def load_checkpoint(
        path: str,
        model: torch.nn.Module,
        optimizers: Dict[str, torch.optim.Optimizer],
        device: Optional[str] = None,
    ) -> Dict[str, Any]:
    """
    Load a model and the optimizer of every strategy from a checkpoint file.

    Both ``model`` and ``optimizers`` must already be constructed (same
    architecture, same optimizers) before calling this function — only
    their state is restored, in place.

    Parameters
    ----------
    path : str
        Path to the checkpoint file written by :func:`save_checkpoint`.
    model : torch.nn.Module
        Model whose parameters are restored in place.
    optimizers : Dict[str, torch.optim.Optimizer]
        Optimizers of every strategy to restore. The keys are the strategy names
        (e.g. ``strategy.name``) and the values are the corresponding optimizer
        objects (e.g. ``strategy.optimizer``).
    device : str, optional
        Device the checkpoint tensors are mapped to. Defaults to the
        device the checkpoint was saved on.

    Returns
    -------
    Dict[str, Any]
        Metadata stored alongside the checkpoint by :func:`save_checkpoint`.
    """
    # ------------------------------------------------------------------------ #
    checkpoint = torch.load(path, map_location=device)

    # ------------------------------------------------------------------------ #
    # Validate that the strategy names match exactly the ones in the checkpoint
    expected = {key for key in optimizers.keys()}
    saved = set(checkpoint["optimizers"].keys())

    if expected != saved:
        raise ValueError(
            f"Strategy names do not match the checkpoint at '{path}': "
            f"expected {sorted(expected)}, found {sorted(saved)}."
        )

    # ------------------------------------------------------------------------ #
    model.load_state_dict(checkpoint["model"])

    for key in optimizers.keys():
        optimizers[key].load_state_dict(checkpoint["optimizers"][key])

    # ------------------------------------------------------------------------ #
    return checkpoint["metadata"]

# -------------------------------------------------------------------------------- #
