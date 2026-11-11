# -------------------------------------------------------------------------------- #
import torch

from collections.abc import Mapping
from numpy.typing import ArrayLike

from sciml.contracts.context import Context
from sciml.interfaces.sampler import SamplerBase

# -------------------------------------------------------------------------------- #
class FixedPoints(SamplerBase):
    """
    Return all prescribed points, in the supplied order, on every evaluation.

    Each call to :meth:`next` returns a new :class:`Context` containing
    the full set of coordinates and targets.

    Examples
    --------
    >>> boundary = FixedPoints({"x": [0, 1], "u_bc": [2, 3]})
    >>> boundary.next()["x"].shape
    torch.Size([2, 1])
    """

    def __init__(
            self,
            points: Mapping[str, ArrayLike | torch.Tensor],
            *,
            device: str = "cpu",
            dtype: torch.dtype = torch.float32,
        ) -> None:
        """
        Parameters
        ----------
        points : Mapping[str, ArrayLike | torch.Tensor]
            Named coordinate and target columns of shape (N,) or (N, 1).
            Entries in the same row describe the same point.
        device : str, default="cpu"
            Device on which the point tensors are stored.
        dtype : torch.dtype, default=torch.float32
            Data type of the point tensors.
        """
        # ------------------------------------------------------------------------ #
        # > Validation
        if not points:
            raise ValueError("Supply at least one named column.")

        # ------------------------------------------------------------------------ #
        # > Prepare fixed columns
        self._columns: dict[str, torch.Tensor] = {}
        num_points = None

        for key, values in points.items():
            column = torch.as_tensor(values, dtype=dtype, device=device)
            if column.ndim == 1:
                column = column.unsqueeze(1)

            if column.ndim != 2 or column.shape[1] != 1 or column.shape[0] == 0:
                raise ValueError(f"Column {key!r} must have shape (N,) or (N, 1) with N > 0.")
            if num_points is None:
                num_points = column.shape[0]
            elif column.shape[0] != num_points:
                raise ValueError(f"Column {key!r} must contain {num_points} points.")
            # > Store a copy independent of the supplied data and its graph
            self._columns[key] = column.detach().clone()

        # ------------------------------------------------------------------------ #
        return

    def next(self) -> Context:
        """
        Return the prescribed points for one evaluation.

        Returns
        -------
        Context
            Evaluation context with an independent (N, 1) tensor per key.
        """
        # ------------------------------------------------------------------------ #
        # > Copy columns so each evaluation can modify its own tensors
        context = Context()
        for key, column in self._columns.items():
            context[key] = column.clone()

        # ------------------------------------------------------------------------ #
        return context

# -------------------------------------------------------------------------------- #
