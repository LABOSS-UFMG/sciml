#####################################################################################
import os
import torch

from pathlib import Path
from typing import Any, Dict, Optional, Union, Mapping, Iterable

from sciml.utils import checker

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

def _process_networks(
        networks: Union[torch.nn.Module, Iterable[torch.nn.Module], Dict[str, torch.nn.Module]]
    ) -> Dict[str, torch.nn.Module]:
    """
    Convert various input formats to a dictionary mapping names to networks.
    
    Parameters
    ----------
    networks : torch.nn.Module or sequence or dict
        Single network, sequence of networks, or dict mapping names to networks.
    
    Returns
    -------
    Dict[str, torch.nn.Module]
        Dictionary mapping names to network instances.
    
    Raises
    ------
    TypeError
        If the input format is not recognized.
    """
    # ---------------------------------------------------------------------------
    # Single network
    if isinstance(networks, torch.nn.Module):
        return {"network": networks}
    
    # Dictionary of networks
    elif isinstance(networks, dict):
        for name, net in networks.items():
            if not isinstance(name, str):
                raise TypeError(f"Network names must be strings, got {type(name)}")
            checker.is_dtype(net, torch.nn.Module)
        return networks
    
    # Sequence of networks - BUT NOT STRINGS (strings are iterable but we want to reject them)
    elif isinstance(networks, Iterable) and not isinstance(networks, str):
        for net in networks:
            checker.is_dtype(net, torch.nn.Module)
        return {f"network_{i}": net for i, net in enumerate(networks)}
    
    else:
        raise TypeError(
            f"network must be a torch.nn.Module, sequence, or dict, got {type(networks)}"
        )


def _process_optimizers(
        optimizers: Union[torch.optim.Optimizer, Iterable[torch.optim.Optimizer], Dict[str, torch.optim.Optimizer]]
    ) -> Dict[str, torch.optim.Optimizer]:
    """
    Convert various input formats to a dictionary mapping names to optimizers.
    
    Parameters
    ----------
    optimizers : torch.optim.Optimizer or sequence or dict
        Single optimizer, sequence of optimizers, or dict mapping names to optimizers.
    
    Returns
    -------
    Dict[str, torch.optim.Optimizer]
        Dictionary mapping names to optimizer instances.
    
    Raises
    ------
    TypeError
        If the input format is not recognized.
    """
    # ---------------------------------------------------------------------------
    # Single optimizer
    if isinstance(optimizers, torch.optim.Optimizer):
        return {"optimizer": optimizers}
    
    # Dictionary of optimizers
    elif isinstance(optimizers, dict):
        for name, opt in optimizers.items():
            if not isinstance(name, str):
                raise TypeError(f"Optimizer names must be strings, got {type(name)}")
            checker.is_dtype(opt, torch.optim.Optimizer)
        return optimizers
    
    # Sequence of optimizers - BUT NOT STRINGS (strings are iterable but we want to reject them)
    elif isinstance(optimizers, Iterable) and not isinstance(optimizers, str):
        for opt in optimizers:
            checker.is_dtype(opt, torch.optim.Optimizer)
        return {f"optimizer_{i}": opt for i, opt in enumerate(optimizers)}
    
    else:
        raise TypeError(
            f"optimizer must be a torch.optim.Optimizer, sequence, or dict, got {type(optimizers)}"
        )


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
    checker.is_dtype(network, torch.nn.Module)
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
    checker.is_dtype(network, torch.nn.Module)

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
    checker.is_dtype(optimizer, torch.optim.Optimizer)
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
    checker.is_dtype(optimizer, torch.optim.Optimizer)

    state_dict = torch.load(path, map_location=map_location)
    optimizer.load_state_dict(state_dict)

    # ---------------------------------------------------------------------------
    return optimizer


#####################################################################################
def save_checkpoint(
        path: PathLike,
        network: Union[torch.nn.Module, Iterable[torch.nn.Module], Dict[str, torch.nn.Module]],
        optimizer: Optional[Union[torch.optim.Optimizer, Iterable[torch.optim.Optimizer], Dict[str, torch.optim.Optimizer]]] = None,
        iteration: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
    """
    Save a full training checkpoint (network(s), optimizer(s), and metadata) to
    a single file.

    This is the function typically used by a ``Checkpoint`` callback: it
    bundles everything needed to resume training later into one file,
    instead of managing separate files for the network and the optimizer.

    Supports multiple networks and optimizers:
    - Single network: pass a single torch.nn.Module
    - Multiple networks: pass a list/tuple of torch.nn.Module or a dict with string keys
    - Single optimizer: pass a single torch.optim.Optimizer
    - Multiple optimizers: pass a list/tuple of torch.optim.Optimizer or a dict with string keys

    Parameters
    ----------
    path : str or Path
        Destination file path. Parent directories are created
        automatically if they don't exist.
    network : torch.nn.Module or sequence or dict
        Network(s) whose parameters will be saved.
    optimizer : torch.optim.Optimizer or sequence or dict, optional
        Optimizer(s) whose state will be saved, if provided.
    iteration : int, optional
        Current iteration number, stored alongside the checkpoint for later
        reference (e.g. to resume training from where it left off).
    metadata : dict, optional
        Any additional information to store with the checkpoint (e.g. the
        best metric value seen so far, the losses at this point, or the
        random seed used).

    Examples
    --------
    >>> # Single network and optimizer
    >>> save_checkpoint(
    ...     "checkpoints/best.pt",
    ...     network=network,
    ...     optimizer=optimizer,
    ...     iteration=120,
    ...     metadata={"best_relative_l2": 0.0031},
    ... )
    >>> 
    >>> # Multiple networks with names
    >>> save_checkpoint(
    ...     "checkpoints/multi.pt",
    ...     network={"encoder": encoder, "decoder": decoder},
    ...     optimizer={"enc_opt": opt1, "dec_opt": opt2},
    ...     iteration=120,
    ... )
    """
    # ---------------------------------------------------------------------------
    # Validate and process networks
    network_dict = _process_networks(network)
    
    # Validate and process optimizers if provided
    optimizer_dict = None
    if optimizer is not None:
        optimizer_dict = _process_optimizers(optimizer)
    
    if iteration is not None:
        checker.is_integer(iteration)
    
    if metadata is not None:
        checker.is_dtype(metadata, Mapping)

    _ensure_parent_dir(path)

    # ---------------------------------------------------------------------------
    checkpoint: Dict[str, Any] = {
        "networks": {name: net.state_dict() for name, net in network_dict.items()}
    }
    
    if optimizer_dict is not None:
        checkpoint["optimizers"] = {
            name: opt.state_dict() for name, opt in optimizer_dict.items()
        }
    
    if iteration is not None:
        checkpoint["iteration"] = iteration

    if metadata is not None:
        checkpoint["metadata"] = metadata

    # ---------------------------------------------------------------------------
    torch.save(checkpoint, path)

    # ---------------------------------------------------------------------------
    return


def load_checkpoint(
        path: PathLike,
        network: Union[torch.nn.Module, Iterable[torch.nn.Module], Dict[str, torch.nn.Module]],
        optimizer: Optional[Union[torch.optim.Optimizer, Iterable[torch.optim.Optimizer], Dict[str, torch.optim.Optimizer]]] = None,
        map_location: Optional[Union[str, torch.device]] = None,
    ) -> Dict[str, Any]:
    """
    Load a full training checkpoint from disk, restoring the network(s) (and
    optimizer(s), if provided) in-place.

    Supports multiple networks and optimizers:
    - Single network: pass a single torch.nn.Module
    - Multiple networks: pass a list/tuple of torch.nn.Module or a dict with string keys
    - Single optimizer: pass a single torch.optim.Optimizer
    - Multiple optimizers: pass a list/tuple of torch.optim.Optimizer or a dict with string keys

    Parameters
    ----------
    path : str or Path
        Path to a file previously created by ``save_checkpoint``.
    network : torch.nn.Module or sequence or dict
        Network instance(s) to load the saved parameters into.
    optimizer : torch.optim.Optimizer or sequence or dict, optional
        Optimizer instance(s) to load the saved state into, if the
        checkpoint contains them. Ignored if the checkpoint has no
        ``"optimizers"`` entry.
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
    >>> # Single network and optimizer
    >>> checkpoint = load_checkpoint(
    ...     "checkpoints/best.pt", network=network, optimizer=optimizer,
    ... )
    >>> start_epoch = checkpoint.get("epoch", 0)
    >>> 
    >>> # Multiple networks with names
    >>> checkpoint = load_checkpoint(
    ...     "checkpoints/multi.pt",
    ...     network={"encoder": encoder, "decoder": decoder},
    ...     optimizer={"enc_opt": opt1, "dec_opt": opt2},
    ... )
    """
    # ---------------------------------------------------------------------------
    # Process networks and optimizers
    network_dict = _process_networks(network)
    
    if optimizer is not None:
        optimizer_dict = _process_optimizers(optimizer)
    else:
        optimizer_dict = None

    # ---------------------------------------------------------------------------
    checkpoint = torch.load(path, map_location=map_location)

    # Load networks
    if "networks" in checkpoint:
        for name, net in network_dict.items():
            if name in checkpoint["networks"]:
                net.load_state_dict(checkpoint["networks"][name])
            else:
                raise KeyError(f"Network '{name}' not found in checkpoint. Available: {list(checkpoint['networks'].keys())}")

    # Load optimizers if provided and checkpoint has them
    if optimizer_dict is not None and "optimizers" in checkpoint:
        for name, opt in optimizer_dict.items():
            if name in checkpoint["optimizers"]:
                opt.load_state_dict(checkpoint["optimizers"][name])
            else:
                raise KeyError(f"Optimizer '{name}' not found in checkpoint. Available: {list(checkpoint['optimizers'].keys())}")

    # ---------------------------------------------------------------------------
    return checkpoint

#####################################################################################
