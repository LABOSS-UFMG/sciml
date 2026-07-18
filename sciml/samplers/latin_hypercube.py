#####################################################################################
import numpy as np
import torch

from typing import Callable, List, Optional, Tuple, Iterable

from scipy.stats import qmc

from sciml.core.sampler import SamplerBase
from sciml.core.context import Context
from sciml.utils import checker

#####################################################################################
class LatinHypercube(SamplerBase):
    """
    Sampler based on Latin Hypercube Sampling (LHS).

    LHS is a stratified sampling technique in which each sampled dimension is
    divided into ``num_points`` equally sized intervals and exactly one sample
    is drawn from each interval. Compared to purely random sampling, this
    produces a more uniform coverage of the sampling domain using the same
    number of points.

    The generated samples are precomputed once during construction and divided
    into batches. Each call to :meth:`next` returns a :class:`Context`
    containing one batch of sampled variables. After the last batch is
    returned, the sampler automatically cycles back to the first one.

    Examples
    --------
    Collocation points ``(x, t)`` sampled over
    ``[0, 1] x [0, 10]``:

    >>> sampler = LatinHypercube(
    ...     dim=2,
    ...     num_points=1000,
    ...     batch_size=100,
    ...     bounds=[(1, (0.0, 10.0))],
    ...     input_keys=["x", "t"],
    ... )
    >>> context = sampler.next()

    Boundary points at ``x = 0`` while sampling only ``t``:

    >>> sampler = LatinHypercube(
    ...     dim=2,
    ...     num_points=1000,
    ...     batch_size=100,
    ...     bounds=[(1, (0.0, 10.0))],
    ...     input_keys=["x", "t"],
    ...     target_keys=["u"],
    ...     insertions=[(0, 0.0)],
    ...     target_fn=lambda x, t: np.zeros((x.shape[0], 1)),
    ... )
    >>> context = sampler.next()
    """

    def __init__(
            self,
            dim: int,
            num_points: int,
            batch_size: int,
            input_keys: Optional[Iterable[str]] = None,
            bounds: Optional[List[Tuple[int, Tuple[float, float]]]] = None,
            insertions: Optional[List[Tuple[int, float]]] = None,
            target_keys: Optional[Iterable[str]] = None,
            target_fn: Optional[Callable[[Iterable[np.ndarray]], Iterable[np.ndarray]]] = None,
            seed: Optional[int] = None,
            device: str = "cpu",
            dtype: torch.dtype = torch.float32,
        ) -> None:
        """
        Parameters
        ----------
        dim : int
            Number of input dimensions.
        num_points : int
            Total number of sampling points.
        batch_size : int
            Number of points returned by each call to :meth:`next`. The last
            batch may contain fewer than ``batch_size`` points if
            ``num_points`` is not an exact multiple of ``batch_size``.
         input_keys : Iterable[str]
            Names used to store the sampled input variables in the returned
            :class:`Context`. The number of keys must equal ``dim``.
        bounds : List[Tuple[int, Tuple[float, float]]], optional
            List of ``(dimension, (lower, upper))`` tuples specifying the
            sampling interval for selected dimensions. Dimensions not listed
            remain sampled over the unit interval ``[0, 1]``.
        insertions : List[Tuple[int, float]], optional
            List of ``(dimension, value)`` tuples specifying dimensions whose
            sampled values are replaced by a constant.
        target_keys : Iterable[str], optional
            Names used to store the target variables returned by
            ``target_fn``.
        target_fn : Callable, optional
            Function used to generate target variables from the sampled input
            points. The function receives one NumPy array for each input
            dimension and returns one NumPy array for each target variable.
        seed : int, optional
            Seed used by the underlying Latin Hypercube generator.
        device : str, default="cpu"
            Device on which the generated tensors are stored.
        dtype : torch.dtype, default=torch.float32
            Data type of the generated tensors.
        """
        # ---------------------------------------------------------------------------
        # > Validation
        checker.is_integer(dim)
        checker.is_integer(num_points)
        checker.is_integer(batch_size)

        if dim <= 0:
            raise "dim must be greater than 0."
        
        if num_points <= 0:
            raise "dim must be greater than 0."
        
        if batch_size <= 0:
            raise "dim must be greater than 0."
        
        if batch_size > num_points:
            raise "num_points must be greater than batch_size."
        
        if not (input_keys is None):
            checker.is_iterable(input_keys, dtype=str)
        
        if not (bounds is None):
            checker.is_iterable(bounds)
            for idx, (lower, upper) in bounds:
                checker.is_integer(idx)
                checker.is_float(lower)
                checker.is_float(upper)
        
        if not (insertions is None):
            checker.is_iterable(insertions)
            for idx, value in insertions:
                checker.is_integer(idx)
                checker.is_float(value)
        
        if not (target_keys is None):
            checker.is_iterable(target_keys, dtype=str)
        
        if not (target_fn is None):
            checker.is_callable(target_fn)
        
        if not (seed is None):
            checker.is_integer(seed)
        
        checker.is_string(device)
        checker.is_dtype(dtype, torch.dtype)

        # ---------------------------------------------------------------------------
        # > Inputs
        self.dim = dim
        self.num_points = num_points
        self.batch_size = batch_size
        self.input_keys = input_keys if not (input_keys is None) else [f"x{i+1}" for i in range(dim)]
        self.bounds = bounds
        self.insertions = insertions
        self.target_keys = target_keys
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

    def _build_batches(self) -> List[List[torch.Tensor]]:
        """
        Generate all sampling points and split them into batches.

        Returns
        -------
        List[List[torch.Tensor | None]]
            Precomputed batches. Each element contains a tensor of sampled
            inputs and, optionally, a tensor of target values.
        """
        # ---------------------------------------------------------------------------
        # > Sample the unit cube via Latin Hypercube Sampling
        engine = qmc.LatinHypercube(d=self.dim, seed=self.seed)
        samples = engine.random(n=self.num_points)

        # > Rescale to `bounds`
        if not (self.bounds is None):
            for i, (low, high) in self.bounds:
                samples[:, i] = low + (high - low) * samples[:, i]

        # > Replace some columns for constant values
        if not (self.insertions is None):
            for i, v in self.insertions:
                samples[:, i] = v

        # > Compute targets (once, over the full NumPy array) if requested
        if not (self.target_fn is None):
            targets = self.target_fn(*[samples[:, i:i+1] for i in range(self.dim)])
        else:
            targets = None

        # > Convert to torch and split into batches
        samples = torch.as_tensor(samples, dtype=self.dtype)
        samples = torch.split(samples, self.batch_size, dim=0)

        if not (targets is None):
            targets = torch.as_tensor(targets, dtype=self.dtype)
            targets = torch.split(targets, self.batch_size, dim=0)

        # ---------------------------------------------------------------------------
        # > Create batches
        batches = []
        for i in range(len(samples)):
            batch = [samples[i], targets[i] if not (targets is None) else None]
            batches.append(batch)

        # ---------------------------------------------------------------------------
        return batches

    def next(self) -> Context:
        """
        Return the next precomputed evaluation context.

        The returned context contains the sampled input variables and,
        optionally, the corresponding target variables. Once the last batch has
        been returned, the sampler automatically cycles back to the first one.

        Returns
        -------
        Context
            Evaluation context containing one batch of sampled variables.
        """
        # ---------------------------------------------------------------------------
        batch = self._batches[self._pointer]

        self._pointer += 1
        if self._pointer >= len(self._batches):
            self._pointer = 0
        
        # ---------------------------------------------------------------------------
        context = Context()
        
        for i, key in enumerate(self.input_keys):
            context[key] = batch[0][:, i:i+1]
        
        if not (batch[1] is None):
            for i, key in enumerate(self.target_keys):
                context[key] = batch[1][:, i:i+1]
        
        # ---------------------------------------------------------------------------
        return context

#####################################################################################
