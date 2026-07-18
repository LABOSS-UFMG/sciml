#####################################################################################
import torch

from typing import Dict

from sciml.utils.autograd import derivative
from sciml.utils import checker

#####################################################################################
class Context():
    """
    Shared container for data associated with a single objective evaluation.

    A ``Context`` stores all information generated during one evaluation of an
    class:`Objective`. It acts as the communication interface between the
    sampler, model and losses.
    """
    def __init__(self):
        # ---------------------------------------------------------------------------
        self._cache: Dict[str, torch.Tensor] = {}    # Data shared during the evaluation
        # ---------------------------------------------------------------------------
        return
    
    def requires_grad(
            self,
            key: str,
        ) -> None:
        """Set requires_grad flag for a variable in the context.

        Parameters
        ----------
        key : str
            Name of the variable in '_cache'.
        """
        # ---------------------------------------------------------------------------
        checker.is_string(key)

        # ---------------------------------------------------------------------------
        self._cache[key].requires_grad_()

        # ---------------------------------------------------------------------------
        return

    def partial(
            self,
            y: str,
            x: str,
            order: int = 1,
        ) -> torch.Tensor:
        """
        Return a partial derivative, computing it only once.

        If the requested derivative has already been computed during the
        current evaluation, the cached value is returned. Otherwise, the
        derivative is computed using automatic differentiation and stored
        for future reuse.

        Parameters
        ----------
        y : str
            Name of the dependent variable in '_cache'.
        x : str
            Name of the independent variable in '_cache'.
        order : int, default=1
            Derivative order.

        Returns
        -------
        Tuple[torch.Tensor, str]
            Requested partial derivative and its key
        """
        # ---------------------------------------------------------------------------
        checker.is_string(y)
        checker.is_string(x)

        if not (x in self._cache):
            raise f"x={x} is not in {self._cache.keys()}"
        
        if not (y in self._cache):
            raise f"y={y} is not in {self._cache.keys()}"

        # ---------------------------------------------------------------------------
        key = self._get_key(y, x, order)

        if key not in self._cache:
            self._cache[key] = derivative(self._cache[y], self._cache[x], order)

        # ---------------------------------------------------------------------------
        return (self._cache[key], key)
    
    def _get_key(
            self,
            y: str,
            x: str,
            order: int,
        ) -> str:
        """
        Build the cache key associated with a partial derivative.

        Parameters
        ----------
        y : str
            Name of the dependent variable.
        x : str
            Name of the independent variable.
        order : int
            Derivative order.

        Returns
        -------
        str
            Unique key used to cache the derivative.
        """
        return f"d{order}{y}_d{x}{order}"

    def __getitem__(
            self,
            key: str,
        ) -> torch.Tensor:
        """
        Retrieve a variable stored in the context.

        Parameters
        ----------
        key : str
            Variable name.

        Returns
        -------
        torch.Tensor
            Stored tensor.
        """
        # ---------------------------------------------------------------------------
        checker.is_string(key)

        # ---------------------------------------------------------------------------
        return self._cache[key]

    def __setitem__(
            self,
            key: str,
            value: torch.Tensor,
        ) -> None:
        """
        Store a variable in the context.

        Parameters
        ----------
        key : str
            Variable name.
        value : torch.Tensor
            Tensor to be stored.
        """
        # ---------------------------------------------------------------------------
        checker.is_string(key)
        checker.is_dtype(value, torch.Tensor)

        # ---------------------------------------------------------------------------
        self._cache[key] = value

        # ---------------------------------------------------------------------------
        return

    def __contains__(
            self,
            key: str,
        ) -> bool:
        """
        Check whether a variable exists in the context.

        Parameters
        ----------
        key : str
            Variable name.

        Returns
        -------
        bool
            ``True`` if the variable exists, otherwise ``False``.
        """
        # ---------------------------------------------------------------------------
        checker.is_string(key)

        # ---------------------------------------------------------------------------
        return key in self._cache

#####################################################################################
