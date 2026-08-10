# -------------------------------------------------------------------------------- #
import torch

from typing import Dict

from sciml.utils.autograd import derivative

# -------------------------------------------------------------------------------- #
class Context():
    """
    Shared container for data associated with a single objective evaluation.

    A ``Context`` stores all information generated during one evaluation of an
    class:`Objective`. It acts as the communication interface between the
    sampler, model and losses.
    """

    def __init__(self) -> None:
        # ------------------------------------------------------------------------ #
        # Data shared during evaluation
        self.cache: Dict[str, torch.Tensor] = {}

        # ------------------------------------------------------------------------ #
        return

    def requires_grad(self, key: str) -> None:
        """Set requires_grad flag for a variable in the context.

        Parameters
        ----------
        key : str
            Name of the variable in 'cache'.
        """
        # ------------------------------------------------------------------------ #
        self.cache[key].requires_grad_()

        # ------------------------------------------------------------------------ #
        return

    def partial(self, y: str, x: str, order: int = 1) -> torch.Tensor:
        """
        Return a partial derivative dy/dx, computing it only once.

        If the requested derivative has already been computed during the
        current evaluation, the cached value is returned. Otherwise, the
        derivative is computed using automatic differentiation and stored
        for future reuse.

        Parameters
        ----------
        y : str
            Name of the dependent variable in 'cache'.
        x : str
            Name of the independent variable in 'cache'.
        order : int, default=1
            Derivative order.

        Returns
        -------
        Tuple[torch.Tensor, str]
            Requested partial derivative and its key
        """
        # ------------------------------------------------------------------------ #
        # Validate input arguments
        if not (x in self.cache):
            raise KeyError(f"x={x} is not in {self.cache.keys()}")

        if not (y in self.cache):
            raise KeyError(f"y={y} is not in {self.cache.keys()}")

        # ------------------------------------------------------------------------ #
        # Compute partial derivative
        key = self._get_key(y, x, order)

        if key not in self.cache:
            self.cache[key] = derivative(self.cache[y], self.cache[x], order)

        # ------------------------------------------------------------------------ #
        return (self.cache[key], key)

    def _get_key(self, y: str, x: str, order: int) -> str:
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
        # ------------------------------------------------------------------------ #
        return f"d{order}{y}_d{x}{order}"

    def __getitem__(self, key: str) -> torch.Tensor:
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
        # ------------------------------------------------------------------------ #
        return self.cache[key]

    def __setitem__(self, key: str, value: torch.Tensor) -> None:
        """
        Store a variable in the context.

        Parameters
        ----------
        key : str
            Variable name.
        value : torch.Tensor
            Tensor to be stored.
        """
        # ------------------------------------------------------------------------ #
        self.cache[key] = value

        # ------------------------------------------------------------------------ #
        return

    def __contains__(self, key: str) -> bool:
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
        # ------------------------------------------------------------------------ #
        return key in self.cache

# -------------------------------------------------------------------------------- #
