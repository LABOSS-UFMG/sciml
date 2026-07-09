#####################################################################################
import os
import torch

from pathlib import Path
from typing import Any, Dict, Optional, Union

from sciml.utils import validation

#####################################################################################
PathLike = Union[str, Path]


def _ensure_parent_dir(path: PathLike) -> None:
    """
    Create the parent directory of ``path``, if it doesn't already exist.

    Parameters
    ----------
    path : str or Path
        File path whose parent directory should exist before writing.
    """
    # ---------------------------------------------------------------------------
    parent = Path(path).parent
    if str(parent) not in ("", "."):
        os.makedirs(parent, exist_ok=True)
    # ---------------------------------------------------------------------------
    return


#####################################################################################
def save_network(network: torch.nn.Module, path: PathLike) -> None:
    """
    Save a network's parameters (state dict) to disk.

    Only the parameters are saved, not the network's architecture. The
    same network class must be instantiated before calling
    ``load_network`` to restore them.

    Parameters
    ----------
    network : torch.nn.Module
        Network whose parameters will be saved.
    path : str or Path
        Destination file path. Parent directories are created
        automatically if they don't exist.

    Examples
    --------
    >>> save_network(network, "checkpoints/network.pt")
    """
    # ---------------------------------------------------------------------------
    validation.is_network(network)
    _ensure_parent_dir(path)

    torch.save(network.state_dict(), path)

    # ---------------------------------------------------------------------------
    return


def load_network(
        network: torch.nn.Module,
        path: PathLike,
        map_location: Optional[Union[str, torch.device]] = None,
    ) -> torch.nn.Module:
    """
    Load a network's parameters (state dict) from disk, in-place.

    The ``network`` instance must already have the same architecture as
    the one whose parameters were saved — only the parameter *values* are
    restored, not the architecture itself.

    Parameters
    ----------
    network : torch.nn.Module
        Network instance to load the saved parameters into.
    path : str or Path
        Path to a file previously created by ``save_network``.
    map_location : str or torch.device, optional
        Device onto which the saved tensors are remapped (e.g. ``"cpu"``
        when loading a GPU-trained network on a machine without a GPU).

    Returns
    -------
    torch.nn.Module
        The same ``network`` instance, with parameters loaded in-place.

    Examples
    --------
    >>> network = load_network(network, "checkpoints/network.pt")
    """
    # ---------------------------------------------------------------------------
    validation.is_network(network)

    state_dict = torch.load(path, map_location=map_location)
    network.load_state_dict(state_dict)

    # ---------------------------------------------------------------------------
    return network


#####################################################################################
def save_optimizer(optimizer: torch.optim.Optimizer, path: PathLike) -> None:
    """
    Save an optimizer's state (state dict) to disk.

    This includes any internal state the optimizer keeps per parameter
    (e.g. momentum buffers in Adam), not just the hyperparameters.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        Optimizer whose state will be saved.
    path : str or Path
        Destination file path. Parent directories are created
        automatically if they don't exist.

    Examples
    --------
    >>> save_optimizer(optimizer, "checkpoints/optimizer.pt")
    """
    # ---------------------------------------------------------------------------
    validation.is_optimizer(optimizer)
    _ensure_parent_dir(path)

    torch.save(optimizer.state_dict(), path)

    # ---------------------------------------------------------------------------
    return


def load_optimizer(
        optimizer: torch.optim.Optimizer,
        path: PathLike,
        map_location: Optional[Union[str, torch.device]] = None,
    ) -> torch.optim.Optimizer:
    """
    Load an optimizer's state (state dict) from disk, in-place.

    The ``optimizer`` instance must already be constructed with the same
    parameter groups as the one whose state was saved.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        Optimizer instance to load the saved state into.
    path : str or Path
        Path to a file previously created by ``save_optimizer``.
    map_location : str or torch.device, optional
        Device onto which the saved tensors are remapped.

    Returns
    -------
    torch.optim.Optimizer
        The same ``optimizer`` instance, with state loaded in-place.

    Examples
    --------
    >>> optimizer = load_optimizer(optimizer, "checkpoints/optimizer.pt")
    """
    # ---------------------------------------------------------------------------
    validation.is_optimizer(optimizer)

    state_dict = torch.load(path, map_location=map_location)
    optimizer.load_state_dict(state_dict)

    # ---------------------------------------------------------------------------
    return optimizer


#####################################################################################
def save_checkpoint(
        path: PathLike,
        network: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        epoch: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
    """
    Save a full training checkpoint (network, optimizer, and metadata) to
    a single file.

    This is the function typically used by a ``Checkpoint`` callback: it
    bundles everything needed to resume training later into one file,
    instead of managing separate files for the network and the optimizer.

    Parameters
    ----------
    path : str or Path
        Destination file path. Parent directories are created
        automatically if they don't exist.
    network : torch.nn.Module
        Network whose parameters will be saved.
    optimizer : torch.optim.Optimizer, optional
        Optimizer whose state will be saved, if provided.
    epoch : int, optional
        Current epoch number, stored alongside the checkpoint for later
        reference (e.g. to resume training from where it left off).
    metadata : dict, optional
        Any additional information to store with the checkpoint (e.g. the
        best metric value seen so far, the losses at this point, or the
        random seed used).

    Examples
    --------
    >>> save_checkpoint(
    ...     "checkpoints/best.pt",
    ...     network=network,
    ...     optimizer=optimizer,
    ...     epoch=120,
    ...     metadata={"best_relative_l2": 0.0031},
    ... )
    """
    # ---------------------------------------------------------------------------
    validation.is_network(network)

    if optimizer is not None:
        validation.is_optimizer(optimizer)
    if epoch is not None:
        validation.is_integer(epoch)
    if metadata is not None:
        validation.is_mapping(metadata)

    _ensure_parent_dir(path)

    # ---------------------------------------------------------------------------
    checkpoint: Dict[str, Any] = {"network": network.state_dict()}
    if optimizer is not None:
        checkpoint["optimizer"] = optimizer.state_dict()
    if epoch is not None:
        checkpoint["epoch"] = epoch
    if metadata is not None:
        checkpoint["metadata"] = metadata

    # ---------------------------------------------------------------------------
    torch.save(checkpoint, path)

    # ---------------------------------------------------------------------------
    return


def load_checkpoint(
        path: PathLike,
        network: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        map_location: Optional[Union[str, torch.device]] = None,
    ) -> Dict[str, Any]:
    """
    Load a full training checkpoint from disk, restoring the network (and
    optimizer, if provided) in-place.

    Parameters
    ----------
    path : str or Path
        Path to a file previously created by ``save_checkpoint``.
    network : torch.nn.Module
        Network instance to load the saved parameters into.
    optimizer : torch.optim.Optimizer, optional
        Optimizer instance to load the saved state into, if the
        checkpoint contains one. Ignored if the checkpoint has no
        ``"optimizer"`` entry.
    map_location : str or torch.device, optional
        Device onto which the saved tensors are remapped.

    Returns
    -------
    Dict[str, Any]
        The raw checkpoint dictionary, including ``"epoch"`` and
        ``"metadata"`` when present, so the caller can resume training
        state beyond just the network and optimizer (e.g. the epoch to
        restart from).

    Examples
    --------
    >>> checkpoint = load_checkpoint(
    ...     "checkpoints/best.pt", network=network, optimizer=optimizer,
    ... )
    >>> start_epoch = checkpoint.get("epoch", 0)
    """
    # ---------------------------------------------------------------------------
    validation.is_network(network)
    
    if optimizer is not None:
        validation.is_optimizer(optimizer)

    # ---------------------------------------------------------------------------
    checkpoint = torch.load(path, map_location=map_location)

    network.load_state_dict(checkpoint["network"])
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    # ---------------------------------------------------------------------------
    return checkpoint

#####################################################################################
