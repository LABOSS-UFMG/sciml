#####################################################################################
import numpy as np
import torch

from typing import Callable, List, Optional, Tuple, Union

from scipy.stats import qmc

from sciml.core.sampler import SamplerBase
from sciml.core.contracts import Batch
from sciml.utils import validation

#####################################################################################
class LatinHypercube(SamplerBase):
    """
    Sampler based on Latin Hypercube Sampling (LHS).

    LHS is a stratified sampling technique: each dimension of the unit cube
    ``[0, 1]^d`` is divided into ``n_points`` equal intervals, and exactly
    one sample falls in each interval per dimension. This produces a much
    more uniform coverage of the domain than plain random sampling, with
    the same number of points — useful for collocation, boundary, and
    initial-condition points alike.

    In addition to standard LHS, this sampler supports:

    - **margin**: shrinking each sampled dimension away from its edges
      (e.g. sampling in ``[eps, 1 - eps]`` instead of ``[0, 1]``), useful
      to avoid placing points exactly on a boundary.
    - **rescaling**: mapping the unit cube to arbitrary ``(low, high)``
      bounds per dimension.
    - **constant insertions**: inserting one or more fixed-value
      dimensions at specific positions in the output, without sampling
      them. This is how a boundary is typically defined: e.g. for a 2D
      spatial + time problem, fixing ``x = 0`` while still sampling ``y``
      and ``t`` freely.

    Note
    ----
    Every batch is precomputed once, at construction time, and stored
    internally. Calling ``next()`` repeatedly cycles through these batches
    in order; once the last one is returned, the internal pointer resets
    to the first batch (the same points are reused, not resampled).

    Examples
    --------
    Collocation points ``(x, t)`` in ``[0, 1] x [0, 1]``, avoiding the
    exact edges, no known targets (used with a residual loss):

    >>> sampler = LatinHypercube(
    ...     n_points=1000,
    ...     batch_size=100,
    ...     bounds=[(0.0, 1.0), (0.0, 1.0)],
    ...     margin=1e-4,
    ...     input_key="xt",
    ... )
    >>> batch = sampler.next()   # Batch(inputs={"xt": tensor[100, 2]}, targets=None)

    Boundary points at ``x = 0``, sampling only ``t``, with a known target
    computed from a NumPy function (e.g. ``u(0, t) = 0``):

    >>> sampler = LatinHypercube(
    ...     n_points=200,
    ...     batch_size=50,
    ...     bounds=[(0.0, 1.0)],              # only t is sampled
    ...     insertions=[(0, 0.0)],            # x = 0 inserted at position 0
    ...     input_key="xt_boundary",
    ...     target_key="u",
    ...     target_fn=lambda xt: np.zeros((xt.shape[0], 1)),
    ... )
    """

    def __init__(
        self,
        name: str,
        n_points: int,
        batch_size: int,
        bounds: List[Tuple[float, float]],
        margin: Union[float, List[float]] = 0.0,
        insertions: Optional[List[Tuple[int, float]]] = None,
        input_key: str = "x",
        target_key: Optional[str] = None,
        target_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        seed: Optional[int] = None,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """
        Parameters
        ----------
        name : str
            Sampler named used to pass to the correct loss.
        n_points : int
            Total number of points to sample.
        batch_size : int
            Number of points returned per call to ``next()``. The last
            batch may be smaller than ``batch_size`` if ``n_points`` is
            not a multiple of it.
        bounds : List[Tuple[float, float]]
            ``(low, high)`` bounds for each *sampled* dimension (i.e. not
            counting any dimension listed in ``insertions``). The number
            of sampled dimensions is ``len(bounds)``.
        margin : float or List[float], default=0.0
            Fraction of each sampled dimension's range to exclude at both
            ends (e.g. ``margin=0.01`` samples in ``[0.01, 0.99]`` instead
            of ``[0, 1]``, before rescaling to ``bounds``). A single float
            applies the same margin to every sampled dimension; a list
            must have one value per sampled dimension. Must satisfy
            ``0 <= margin < 0.5``.
        insertions : List[Tuple[int, float]], optional
            List of ``(position, value)`` pairs. Each inserts a constant
            column with the given ``value`` at ``position`` (0-indexed) in
            the *final* output (i.e. after combining with the sampled
            dimensions). Positions refer to the output array, which has
            ``len(bounds) + len(insertions)`` columns in total.
        input_key : str, default="x"
            Key used to store the sampled points in ``batch.inputs``.
        target_key : str, optional
            Key used to store the computed targets in ``batch.targets``.
            Required if ``target_fn`` is provided.
        target_fn : Callable[[np.ndarray], np.ndarray], optional
            Function that receives the full sampled array as a NumPy
            array (shape ``[n_points, total_dims]``, i.e. already
            including any inserted constant columns) and returns a NumPy
            array of targets (shape ``[n_points, ...]``), used to populate
            ``batch.targets[target_key]``. If ``None``, ``batch.targets``
            is ``None`` for every batch (typical for collocation points).
        seed : int, optional
            Seed used by the underlying LHS generator, for reproducibility.
        device : str, default="cpu"
            Device on which the generated tensors are stored.
        dtype : torch.dtype, default=torch.float32
            Data type of the generated tensors.
        """
        # ---------------------------------------------------------------------------
        # > Validation
        validation.is_string(name)
        validation.is_integer(n_points)
        validation.is_integer(batch_size)
        validation.is_string(input_key)
        validation.is_string(device)
        validation.is_dtype(dtype, torch.dtype)

        if n_points <= 0:
            raise ValueError("n_points must be a positive integer")
        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if not bounds:
            raise ValueError("bounds must contain at least one (low, high) pair")

        n_sampled_dims = len(bounds)
        insertions = insertions or []
        total_dims = n_sampled_dims + len(insertions)

        insertion_positions = {pos: value for pos, value in insertions}
        if len(insertion_positions) != len(insertions):
            raise ValueError("insertions contain duplicate positions")
        for pos in insertion_positions:
            if not (0 <= pos < total_dims):
                raise ValueError(
                    f"insertion position {pos} out of range for total_dims={total_dims}"
                )

        if isinstance(margin, (int, float)):
            margins = [float(margin)] * n_sampled_dims
        else:
            margins = list(margin)
            if len(margins) != n_sampled_dims:
                raise ValueError("margin list must have one value per sampled dimension")
        for eps in margins:
            if not (0.0 <= eps < 0.5):
                raise ValueError("each margin value must satisfy 0 <= margin < 0.5")

        if target_fn is not None:
            validation.is_callable(target_fn)
            if target_key is None:
                raise ValueError("target_key must be provided when target_fn is set")
            validation.is_string(target_key)

        if seed is not None:
            validation.is_integer(seed)

        # ---------------------------------------------------------------------------
        # > Inputs
        self.name = name
        self.n_points = n_points
        self.batch_size = batch_size
        self.bounds = bounds
        self.margins = margins
        self.insertion_positions = insertion_positions
        self.total_dims = total_dims
        self.input_key = input_key
        self.target_key = target_key
        self.target_fn = target_fn
        self.seed = seed
        self.device = device
        self.dtype = dtype

        # ---------------------------------------------------------------------------
        # > Precompute batches
        self._batches = self._build_batches()
        self._pointer = 0

        # ---------------------------------------------------------------------------
        return

    def _build_batches(self) -> List[Batch]:
        """
        Generate the full set of LHS points and split them into batches.

        Returns
        -------
        List[Batch]
            The precomputed list of batches, cycled through by ``next()``.
        """
        # ---------------------------------------------------------------------------
        # > Sample the unit cube via Latin Hypercube Sampling
        n_sampled_dims = len(self.bounds)
        engine = qmc.LatinHypercube(d=n_sampled_dims, seed=self.seed)
        unit_samples = engine.random(n=self.n_points)  # shape [n_points, n_sampled_dims]

        # > Apply margin and rescale to `bounds`
        scaled = np.empty_like(unit_samples)
        for i, (low, high) in enumerate(self.bounds):
            eps = self.margins[i]
            u = eps + (1.0 - 2.0 * eps) * unit_samples[:, i]
            scaled[:, i] = low + (high - low) * u

        # > Assemble the final array, inserting constant columns
        full = np.empty((self.n_points, self.total_dims), dtype=np.float64)
        sampled_idx = 0
        for col in range(self.total_dims):
            if col in self.insertion_positions:
                full[:, col] = self.insertion_positions[col]
            else:
                full[:, col] = scaled[:, sampled_idx]
                sampled_idx += 1

        # > Compute targets (once, over the full NumPy array) if requested
        targets_full = self.target_fn(full) if self.target_fn is not None else None

        # > Convert to torch and split into batches
        x_full = torch.as_tensor(full, dtype=self.dtype)
        x_chunks = torch.split(x_full, self.batch_size, dim=0)

        if targets_full is not None:
            target_full_tensor = torch.as_tensor(targets_full, dtype=self.dtype)
            target_chunks = torch.split(target_full_tensor, self.batch_size, dim=0)
        else:
            target_chunks = None

        # ---------------------------------------------------------------------------
        batches = []
        for i, x_chunk in enumerate(x_chunks):
            if target_chunks is not None:
                targets = {self.target_key: target_chunks[i].to(self.device)}
            else:
                targets = None
            batches.append(Batch(inputs={self.input_key: x_chunk.to(device=self.device)}, targets=targets))

        # ---------------------------------------------------------------------------
        return batches

    def next(self) -> Batch:
        """
        Return the next precomputed batch, cycling back to the first one
        once the last batch has been returned.

        Returns
        -------
        Batch
            The next batch of sampled points.
        """
        # ---------------------------------------------------------------------------
        batch = self._batches[self._pointer]

        self._pointer += 1
        if self._pointer >= len(self._batches):
            self._pointer = 0

        # ---------------------------------------------------------------------------
        return batch

#####################################################################################
