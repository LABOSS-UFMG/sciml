#####################################################################################
import torch

from typing import Dict, List, Optional

from sciml.core.trainer import TrainerBase
from sciml.core.loss import LossBase
from sciml.core.sampler import SamplerBase
from sciml.core.metric import MetricBase
from sciml.core.callback import CallbackBase
from sciml.callbacks.checkpoint import Checkpoint
from sciml.callbacks.logger import Logger
from sciml.utils import validation

#####################################################################################
class Staged(TrainerBase):
    """
    The Staged Trainer enables the use of two (or more) optimizers during
    training.

    This trainer works, unmodified, with **any** ``torch.optim.Optimizer``
    — including both first-order optimizers (Adam, SGD, ...) and
    quasi-Newton optimizers that require multiple function re-evaluations
    per step (L-BFGS). It achieves this by always building a `closure`
    (a function that zeroes gradients, evaluates every loss, and calls
    ``backward()``) and calling ``optimizer.step(closure)``:

    - first-order optimizers call ``closure`` exactly once per step;
    - L-BFGS calls ``closure`` multiple times per step, to perform its
      internal line search.

    Because of this, the trainer's code never branches on the type of
    optimizer being used.

    Note
    ----
    Since the closure may be called more than once per iteration (under
    L-BFGS), only the loss/gradient computation lives inside it.
    Everything that should happen exactly once per iteration — advancing
    samplers, evaluating validation metrics, and calling callbacks — lives
    outside the closure, in the main loop.

    Losses and samplers are matched to each other by name: every
    ``LossBase.name`` in ``losses`` must have a corresponding
    ``SamplerBase.name`` in ``samplers``.

    All problem components (network, losses, samplers, optimizer,
    metrics, callbacks) are provided once, at construction time.
    ``fit(iterations)`` can then be called as many times as needed
    (e.g. to train in stages, or to keep extending training after
    inspecting intermediate results) without repeating any of them — the
    internal iteration counter continues from where the previous call
    left off.

    Examples
    --------
    >>> trainer = Staged(
    ...     network=network,
    ...     losses=[residual_loss, boundary_loss],
    ...     samplers=[residual_sampler, boundary_sampler],
    ...     optimizers={
    ...         "adam": torch.optim.Adam(network.parameters()),
    ...         "lbfgs": torch.optim.LBFGS(network.parameters()),
    ...     },
    ...     metrics=[relative_l2],
    ...     validation_sampler=validation_sampler,
    ...     checkpoints=[checkpoint],
    ...     logger=logger,
    ... )
    >>> trainer.fit(iterations=5000, verbose=True)
    >>> # later, continue training with a different optimizer:
    >>> trainer.optimizer = torch.optim.LBFGS(network.parameters())
    >>> trainer.fit(iterations=500, verbose=True)
    """

    def __init__(
            self,
            network: torch.nn.Module,
            losses: List[LossBase],
            samplers: List[SamplerBase],
            optimizers: Dict[str, torch.optim.Optimizer],
            metrics: Optional[List[MetricBase]] = None,
            validation_sampler: Optional[SamplerBase] = None,
            validation_frequency: int = 1,
            checkpoints: Optional[List[Checkpoint]] = None,
            logger: Optional[Logger] = None,
            schedulers: Optional[List[CallbackBase]] = None,
        ) -> None:
        """
        Parameters
        ----------
        network : torch.nn.Module
            Network to be trained.
        losses : List[LossBase]
            Loss terms to optimize. Each must have a unique ``name``,
            matching exactly one entry in ``samplers``.
        samplers : List[SamplerBase]
            Samplers providing the batches for each loss. Each must have
            a unique ``name``, matching exactly one entry in ``losses``.
        optimizers : Dict[str, torch.optim.Optimizer]
            Optimizers used to update ``network``'s parameters. Works with
            any PyTorch optimizer, including L-BFGS.
        metrics : List[MetricBase], optional
            Validation metrics, evaluated against ``validation_sampler``.
            Ignored if ``validation_sampler`` is not provided.
        validation_sampler : SamplerBase, optional
            Dedicated sampler providing validation batches (with known
            targets). If ``None``, no validation is performed and
            ``metrics`` is ignored.
        validation_frequency : int, default=1
            Evaluate ``metrics`` every this many iterations.
        checkpoints : List[Checkpoint], optional
            Checkpoint callbacks, saving the network/optimizer to disk.
        logger : Logger, optional
            Logger callback, recording losses/metrics/network info.
        schedulers : List[CallbackBase], optional
            Parameterless callbacks that mutate arbitrary training state
            (e.g. loss weights) on their own schedule. Their hooks are
            called with no arguments.
        """
        # ---------------------------------------------------------------------------
        # > Validation
        validation.is_network(network)
        validation.is_iterable(losses, dtype=LossBase)
        validation.is_iterable(samplers, dtype=SamplerBase)

        if metrics is not None:
            validation.is_iterable(metrics, dtype=MetricBase)
        
        validation.is_integer(validation_frequency)
        if validation_frequency <= 0:
            raise ValueError("validation_frequency must be a positive integer")
        
        if checkpoints is not None:
            validation.is_iterable(checkpoints, dtype=Checkpoint)
        
        validation.is_dtype(logger, Logger)

        if schedulers is not None:
            validation.is_iterable(schedulers, dtype=CallbackBase)

        losses_by_name: Dict[str, LossBase] = {loss.name: loss for loss in losses}
        samplers_by_name: Dict[str, SamplerBase] = {s.name: s for s in samplers}

        if set(losses_by_name) != set(samplers_by_name):
            raise ValueError(
                f"loss names {sorted(losses_by_name)} do not match "
                f"sampler names {sorted(samplers_by_name)}"
            )

        # ---------------------------------------------------------------------------
        # > Inputs
        self.network = network
        self.losses_by_name = losses_by_name
        self.samplers_by_name = samplers_by_name
        self.optimizers = optimizers
        self.validation_sampler = validation_sampler
        self.validation_frequency = validation_frequency
        self.checkpoints = checkpoints or []
        self.logger = logger
        self.schedulers = schedulers or []

        # ---------------------------------------------------------------------------
        # > Internal parameters
        self.metrics_by_name: Dict[str, MetricBase] = (
            {m.name: m for m in metrics} if metrics is not None else {}
        )
        self._iteration = 0
        self._started = False

        # ---------------------------------------------------------------------------
        return

    def fit(
            self,
            iterations: int,
            optimizer: str,
            verbose: bool = False,
        ) -> None:
        """
        Train the network for ``iterations`` additional iterations.

        Can be called multiple times; the internal iteration counter
        continues from where the previous call left off (e.g. calling
        ``fit(1000)`` twice trains for 2000 iterations total, with
        ``Checkpoint``/``Logger`` frequencies counted against the global
        iteration number, not restarted at each call).

        Parameters
        ----------
        iterations : int
            Number of additional iterations to train for.
        optimizer: str
            Optimizer name used to update ``network``'s parameters.
        verbose : bool, default=False
            If ``True``, print the current loss values every iteration,
            and print any exception raised during training.
        
        Raises
        ------
        Exception
            Any exception raised during training is caught, reported to
            every callback's ``on_exception`` hook, optionally printed
            (if ``verbose``), and then re-raised.
        """
        # ---------------------------------------------------------------------------
        # > Validate
        validation.is_integer(iterations)
        
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")

        validation.is_string(optimizer)

        if not (optimizer in self.optimizers.keys()):
            raise ValueError(f"optimizer must be in {self.optimizers.keys()}")

        validation.is_boolean(verbose)

        # ---------------------------------------------------------------------------
        # > Set optimizer
        optim = self.optimizers[optimizer]

        # ---------------------------------------------------------------------------
        # > Notify callbacks: training start (only once, on the first call)
        if not self._started:
            for callback in self.checkpoints:
                callback.on_train_start()
            
            if self.logger:
                self.logger.on_train_start(network=self.network, optimizers=self.optimizers.values())
            
            for callback in self.schedulers:
                callback.on_train_start()
            
            self._started = True

        # ---------------------------------------------------------------------------
        try:

            for _ in range(iterations):
                self._iteration += 1
        
                # -------------------------------------------------------------------
                # > Notify callbacks
                for callback in self.checkpoints:
                    callback.on_iteration_start()

                if self.logger:
                    self.logger.on_iteration_start()

                for callback in self.schedulers:
                    callback.on_iteration_start()

                # -------------------------------------------------------------------
                # > Sample this iteration's batches once
                batches = {
                    name: sampler.next()
                    for name, sampler in self.samplers_by_name.items()
                }

                # -------------------------------------------------------------------
                # > Compute loss
                loss_values: Dict[str, float] = {}

                def closure():
                    optim.zero_grad()
                    total = 0.0
                    for name, loss_fn in self.losses_by_name.items():
                        value = loss_fn.evaluate(self.network, batches[name])
                        loss_values[name] = value.item()
                        total = total + loss_fn.weight * value
                    total.backward()
                    return total

                total_loss = optim.step(closure)

                # -------------------------------------------------------------------
                # > Validation (only every `validation_frequency` iterations)
                metric_values: Optional[Dict[str, float]] = None

                check_1 = self.validation_sampler is not None
                check_2 = self._iteration % self.validation_frequency == 0
                if check_1 and check_2:
                    validation_batch = self.validation_sampler.next()
                    x = next(iter(validation_batch.inputs.values()))
                    y_true = next(iter(validation_batch.targets.values()))
                    predictions = self.network(x)
                    metric_values = {
                        name: metric.evaluate(predictions, y_true).item()
                        for name, metric in self.metrics_by_name.items()
                    }

                # -------------------------------------------------------------------
                # > Show progress
                if verbose:
                    width = 14

                    def format_header(text: str) -> str:
                        text = str(text)
                        if len(text) > width:
                            text = text[:width - 1] + "."
                        return f"{text:>{width}}"

                    def format_value(value) -> str:
                        if value is None:
                            return f"{'-':>{width}}"

                        if isinstance(value, torch.Tensor):
                            value = value.detach().cpu().item()

                        return f"{float(value):>{width}.6e}"

                    headers = (
                        ["iteration", "total"]
                        + list(loss_values.keys())
                        + list(self.metrics_by_name.keys())
                    )

                    values = (
                        [self._iteration, total_loss]
                        + [loss_values[name] for name in loss_values.keys()]
                        + [
                            metric_values[name]
                            if metric_values is not None and name in metric_values
                            else None
                            for name in self.metrics_by_name.keys()
                        ]
                    )

                    if self._iteration == 1:
                        header = " ".join(format_header(name) for name in headers)
                        separator = " ".join("-" * width for _ in headers)

                        print(header)
                        print(separator)

                    row = " ".join(
                        f"{value:>{width}d}" if name == "iteration" else format_value(value)
                        for name, value in zip(headers, values)
                    )

                    print(row)

                # -------------------------------------------------------------------
                # > Notify callbacks
                for callback in self.checkpoints:
                    callback.on_iteration_end(
                        network=self.network,
                        optimizer=optim,
                        iteration=self._iteration,
                    )

                if self.logger:
                    self.logger.on_iteration_end(
                        iteration=self._iteration,
                        losses=loss_values,
                        metrics=metric_values,
                    )

                for callback in self.schedulers:
                    callback.on_iteration_end()

        # ---------------------------------------------------------------------------
        except Exception as exception:

            # -----------------------------------------------------------------------
            # > Notify callbacks
            for callback in self.checkpoints:
                callback.on_exception(
                    network=self.network,
                    optimizer=optim,
                    iteration=self._iteration,
                )
            
            if self.logger:
                self.logger.on_exception(
                    iteration=self._iteration,
                    exception=exception,
                )

            for callback in self.schedulers:
                callback.on_exception()

            # -----------------------------------------------------------------------
            # > Show exception
            if verbose:
                print(f"[{self._iteration}] Exception raised: {exception!r}")

            raise

        # ---------------------------------------------------------------------------
        # > Notify callbacks: training end (every successful call to fit)
        for callback in self.checkpoints:
            callback.on_train_end(
                network=self.network,
                optimizer=optim,
                iteration=self._iteration,
            )

        if self.logger:
            self.logger.on_train_end()

        for callback in self.schedulers:
            callback.on_train_end()

        # ---------------------------------------------------------------------------
        return

#####################################################################################
