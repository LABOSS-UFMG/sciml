#####################################################################################
import torch

from collections.abc import Mapping, Iterable
from typing import Any, Iterable

from sciml.core.metric import MetricBase

#####################################################################################
def is_integer(x: Any) -> None:
    """
    Validate that the input is a integer.

    Parameters
    ----------
    x : Any
        Object expected to be of type ``int``.

    Raises
    ------
    TypeError
        If ``x`` is not an integer.
    """
    # ---------------------------------------------------------------------------
    if not isinstance(x, int):
        raise TypeError("Input must be of type int")
    # ---------------------------------------------------------------------------
    return


def is_float(x: Any) -> None:
    """
    Validate that the input is a float.

    Parameters
    ----------
    x : Any
        Object expected to be of type ``float``.

    Raises
    ------
    TypeError
        If ``x`` is not an float.
    """
    # ---------------------------------------------------------------------------
    if not isinstance(x, float):
        raise TypeError("Input must be of type float")
    # ---------------------------------------------------------------------------
    return


def is_string(x: Any) -> None:
    """
    Validate that the input is a string.

    Parameters
    ----------
    x : Any
        Object expected to be of type ``str``.

    Raises
    ------
    TypeError
        If ``x`` is not a string.
    """
    # ---------------------------------------------------------------------------
    if not isinstance(x, str):
        raise TypeError("Input must be of type str")
    # ---------------------------------------------------------------------------
    return


def is_boolean(x: Any) -> None:
    """
    Validate that the input is a boolean.

    Parameters
    ----------
    x : Any
        Object expected to be of type ``bool``.

    Raises
    ------
    TypeError
        If ``x`` is not a boolean.
    """
    # ---------------------------------------------------------------------------
    if not isinstance(x, bool):
        raise TypeError("Input must be of type bool")
    # ---------------------------------------------------------------------------
    return


def is_callable(x: Any) -> None:
    """
    Validate that the input is callable with optional argument and
    return-type constraints.

    Parameters
    ----------
    x : Any
        Object expected to be callable.

    Raises
    ------
    TypeError
        If ``x`` is not callable.
    """
    # ---------------------------------------------------------------------------
    if not callable(x):
        raise TypeError("Input must be callable")
    # ---------------------------------------------------------------------------
    return


def is_iterable(x: Any, *, dtype: Any = None) -> None:
    """
    Validate that the input is an iterable with optional length and
    element-type constraints.

    Parameters
    ----------
    x : Any
        Object expected to implement the ``Iterable`` interface.

    dtype : Type[Any], optional
        Expected type for each element of the iterable. If ``None``, no
        element-type constraint is enforced.

    Raises
    ------
    TypeError
        If ``x`` is not an iterable or if any element does not match
        ``dtype``.
            
    ValueError
        If ``length`` is enforced and the iterable length does not match.
    """
    # ---------------------------------------------------------------------------
    # Check iterable contract
    if not isinstance(x, Iterable):
        raise TypeError("Input must be of type Iterable")

    # Enforce element dtype
    if not (dtype is None):
        for i, item in enumerate(x):
            if not isinstance(item, dtype):
                raise TypeError(f"Iterable element at index {i} must be of type {dtype}")
        
    # ---------------------------------------------------------------------------
    return


def is_mapping(x: Any, *, keys: Iterable[Any] = None, dtype: Any = None) -> None:
    """
    Validate that the input is a mapping with optional key and value-type
    constraints.

    Parameters
    ----------
    x : Any
        Object expected to implement the ``Mapping`` interface.

    keys : Iterable[Any], optional
        Keys that must be present in the mapping. If ``None``, no key
        constraint is enforced.

    dtype : Any, optional
        Expected type for the values associated with ``keys``. If ``None``,
        no value-type constraint is enforced.

    Raises
    ------
    TypeError
        If ``x`` is not a mapping, or if any required value does not match
        ``dtype``.

    KeyError
        If any required key is missing from the mapping.
    """
    # ---------------------------------------------------------------------------
    # Check if input x is a mapping
    if not isinstance(x, Mapping):
        raise TypeError("Input must be of type Mapping")
        
    # Enforce a minimal contract on dataset
    if not (keys is None):
        for key in keys:
            if not (key in x):
                raise KeyError(f"Mapping must contain {key} as key")
        
    # Explicit tensor checks
    if not (dtype is None):
        for key in keys:
            if not isinstance(x[key], dtype):
                raise TypeError(f"Mapping content must be {dtype} instances")
        
    # ---------------------------------------------------------------------------
    return
    

def is_metric(x: Any) -> None:
    """
    Validate that the input is a metric instance.

    Parameters
    ----------
    x : Any
        Object expected to be an instance of ``MetricBase``.

    Raises
    ------
    TypeError
        If ``x`` is not an instance of ``MetricBase``.
    """
    # ---------------------------------------------------------------------------
    if not isinstance(x, MetricBase):
        raise TypeError("Input must be an instance of MetricBase")
    # ---------------------------------------------------------------------------
    return


def is_network(x: Any) -> None:
    """
    Validate that the input is a PyTorch module.

    Parameters
    ----------
    x : Any
        Object expected to be an instance of ``torch.nn.Module``.

    Raises
    ------
    TypeError
        If ``x`` is not a ``torch.nn.Module`` instance.
    """
    # ---------------------------------------------------------------------------
    if not isinstance(x, torch.nn.Module):
        raise TypeError("Input must be an instance of torch.nn.Module")
    # ---------------------------------------------------------------------------
    return
    
    
def is_optimizer(x: Any) -> None:
    """
    Validate that the input is a PyTorch Optimizer.

    Parameters
    ----------
    x : Any
        Object expected to be an instance of ``torch.optim.Optimizer``.

    Raises
    ------
    TypeError
        If ``x`` is not a ``torch.optim.Optimizer`` instance.
    """
    # ---------------------------------------------------------------------------
    if not isinstance(x, torch.optim.Optimizer):
        raise TypeError("Input must be an instance of torch.optim.Optimizer")
    # ---------------------------------------------------------------------------
    return


def is_dtype(x: Any, dtype: Any) -> None:
    """
    Validate that the input is specific dtype.

    Parameters
    ----------
    x : Any
        Object expected to be an instance of dtype.
    dtype : Any
        Expected type for the values associated with ``dtype``.
    

    Raises
    ------
    TypeError
        If ``x`` is not an instance of dtype.
    """
    # ---------------------------------------------------------------------------
    if not isinstance(x, dtype):
        raise TypeError(f"Input must be an instance of {dtype}")
    # ---------------------------------------------------------------------------
    return
    
#####################################################################################
