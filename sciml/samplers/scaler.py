#####################################################################################
from typing import Sequence, Union

import numpy as np
import torch

#####################################################################################
Array = Union[np.ndarray, torch.Tensor]

#####################################################################################
class Scaler:
    """
    Linear scaler from the normalized domain [0, 1] to a real domain.

    Examples
    --------
    >>> scaler = Scaler(bounds=[(2.0, 4.0)])
    >>> x = torch.tensor([[0.0], [0.5], [1.0]])
    >>> scaler.transform(x)
    tensor([[2.],
            [3.],
            [4.]])

    >>> scaler = Scaler(bounds=[(0.0, 10.0), (-1.0, 1.0)])
    >>> x = np.array([[0.5, 0.5]])
    >>> scaler.transform(x)
    array([[5., 0.]])
    """

    def __init__(
            self,
            bounds: Sequence[tuple[float, float]],
        ) -> None:
        """
        Parameters
        ----------
        bounds : Sequence[tuple[float, float]]
            Bounds of each dimension. For example:

            - 1D: [(0.0, 2.0)]
            - 2D: [(0.0, 1.0), (-1.0, 1.0)]
        """

        self.bounds = torch.as_tensor(bounds, dtype=torch.float32)
        self.dim = len(self.bounds)

        if self.bounds.ndim != 2 or self.bounds.shape[1] != 2:
            raise ValueError("bounds must have shape (dim, 2).")

        # ---------------------------------------------------------------------------
        lower = self.bounds[:, 0]
        upper = self.bounds[:, 1]

        # ---------------------------------------------------------------------------
        if torch.any(upper <= lower):
            raise ValueError("Each upper bound must be greater than the lower bound.")

        # ---------------------------------------------------------------------------
        self.lower = lower
        self.upper = upper

        # ---------------------------------------------------------------------------
        return

    def transform(self, x: Array) -> Array:
        """
        Transform values from [0, 1] to the real domain.

        Parameters
        ----------
        x : np.ndarray or torch.Tensor
            Normalized values. The last dimension must match the number
            of bounds.

        Returns
        -------
        np.ndarray or torch.Tensor
            Values transformed to the real domain. The output type follows
            the input type.
        """

        # ---------------------------------------------------------------------------
        if isinstance(x, torch.Tensor):
            return self._transform_torch(x)

        if isinstance(x, np.ndarray):
            return self._transform_numpy(x)

        raise TypeError("x must be either a torch.Tensor or a numpy.ndarray.")

    def _transform_torch(self, x: torch.Tensor) -> torch.Tensor:
        """
        Transform torch tensor values from [0, 1] to the real domain.
        """

        # ---------------------------------------------------------------------------
        if x.shape[-1] != self.dim:
            raise ValueError(
                f"The last dimension of x must be {self.dim}, "
                f"but got {x.shape[-1]}."
            )

        # ---------------------------------------------------------------------------
        dtype = x.dtype

        lower = self.lower.to(device=x.device, dtype=dtype)
        upper = self.upper.to(device=x.device, dtype=dtype)

        # ---------------------------------------------------------------------------
        return lower + x * (upper - lower)

    def _transform_numpy(self, x: np.ndarray) -> np.ndarray:
        """
        Transform numpy array values from [0, 1] to the real domain.
        """

        # ---------------------------------------------------------------------------
        if x.shape[-1] != self.dim:
            raise ValueError(
                f"The last dimension of x must be {self.dim}, "
                f"but got {x.shape[-1]}."
            )

        # ---------------------------------------------------------------------------
        dtype = x.dtype

        lower = self.lower.detach().cpu().numpy().astype(dtype)
        upper = self.upper.detach().cpu().numpy().astype(dtype)

        # ---------------------------------------------------------------------------
        return lower + x * (upper - lower)

#####################################################################################
