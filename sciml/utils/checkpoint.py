# -------------------------------------------------------------------------------- #
import torch

from typing import Any, Dict, Optional, Sequence

from sciml.core.strategy import Strategy

# -------------------------------------------------------------------------------- #
def save_checkpoint(
        path: str,
        model: torch.nn.Module,
        strategies: Sequence[Strategy],
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
    strategies : Sequence[Strategy]
        Strategies whose optimizers are saved, keyed by ``strategy.name``.
    metadata : Dict[str, Any], optional
        Arbitrary extra information to store alongside the checkpoint
        (e.g. iteration count, random seed, problem parameters).
    """
    # ------------------------------------------------------------------------ #
    checkpoint = {
        "model": model.state_dict(),
        "optimizers": {
            strategy.name: strategy.optimizer.state_dict() for strategy in strategies
        },
        "metadata": metadata if (metadata is not None) else {},
    }

    torch.save(checkpoint, path)

    # ------------------------------------------------------------------------ #
    return

def load_checkpoint(
        path: str,
        model: torch.nn.Module,
        strategies: Sequence[Strategy],
        device: Optional[str] = None,
    ) -> Dict[str, Any]:
    """
    Load a model and the optimizer of every strategy from a checkpoint file.

    Both ``model`` and ``strategies`` must already be constructed (same
    architecture, same optimizers) before calling this function — only
    their state is restored, in place.

    Parameters
    ----------
    path : str
        Path to the checkpoint file written by :func:`save_checkpoint`.
    model : torch.nn.Module
        Model whose parameters are restored in place.
    strategies : Sequence[Strategy]
        Strategies whose optimizers are restored in place. Their names
        must match exactly the strategy names stored in the checkpoint.
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
    expected = {strategy.name for strategy in strategies}
    saved = set(checkpoint["optimizers"].keys())

    if expected != saved:
        raise ValueError(
            f"Strategy names do not match the checkpoint at '{path}': "
            f"expected {sorted(expected)}, found {sorted(saved)}."
        )

    # ------------------------------------------------------------------------ #
    model.load_state_dict(checkpoint["model"])

    for strategy in strategies:
        strategy.optimizer.load_state_dict(checkpoint["optimizers"][strategy.name])

    # ------------------------------------------------------------------------ #
    return checkpoint["metadata"]

# -------------------------------------------------------------------------------- #
