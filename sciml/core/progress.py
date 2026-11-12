# -------------------------------------------------------------------------------- #
import csv
import time

from pathlib import Path
from shutil import copyfileobj
from tempfile import TemporaryDirectory
from typing import Dict, Sequence

from sciml.contracts.dataclasses import Step
from sciml.core.strategy import Strategy

# -------------------------------------------------------------------------------- #
class Progress():
    """
    Track and report the evolution of the objective function during training.

    A ``Progress`` is built from the full set of strategies used by the
    trainer, so that the printed layout has one fixed line per strategy,
    regardless of whether that strategy is active at a given iteration. For
    each strategy it keeps the initial objective value and the latest one,
    and exposes a live, multi-line rendering meant to be redrawn in place
    every iteration. It can also persist a tabular history of each
    strategy to its own CSV file via :meth:`save`.

    Note
    ----
    A strategy that is disabled at a given iteration (see ``Strategy.enable``)
    does not produce a ``Step``. Its line keeps showing the last known
    values, tagged as inactive, rather than being treated as zero or
    removed from the display.
    """

    def __init__(
            self,
            strategies: Sequence[Strategy],
            verbose: bool = False,
            interval: int = 100,
            path: str | Path | None = None,
            filename: str = "log.csv",
        ) -> None:
        """
        Parameters
        ----------
        strategies : Sequence[Strategy]
            Strategies tracked by this progress report. Their order fixes
            the order of the printed lines.
        verbose : bool, default=False
            Display progress during training.
        interval : int, default=100
            Display every interval global iterations, on iteration one and
            on the last iteration of each fit call.
        path : str or Path, optional
            Output directory. Omit to track progress without writing files.
        filename : str, default="log.csv"
            Base filename, prefixed with the strategy name or "validation".
        """
        # ------------------------------------------------------------------------ #
        # Store constructor arguments
        if type(interval) is not int or interval <= 0:
            raise ValueError("interval must be a positive integer.")
        if not filename or Path(filename).name != filename:
            raise ValueError("filename must be a filename without a directory.")
        self.path = Path(path) if path is not None else None
        self.filename = filename
        self._pending: dict[str, list[dict[str, int | float]]] = {}
        self.verbose = verbose
        self.interval = interval
        self.names = [strategy.name for strategy in strategies]

        # ------------------------------------------------------------------------ #
        # Internal parameters
        self.initial: Dict[str, float] = {}    # First value observed for each strategy
        self.current: Dict[str, float] = {}    # Last value observed for each strategy
        self.active: Dict[str, bool] = {name: False for name in self.names}

        self._start_time = time.perf_counter()
        self._rendered = False

        # ------------------------------------------------------------------------ #
        return

    def update(self, steps: Sequence[Step], iteration: int) -> None:
        """
        Update the tracked statistics with the steps of the current iteration.

        Parameters
        ----------
        steps : Sequence[Step]
            Steps returned by the strategies at the current iteration. A
            strategy that did not produce a ``Step`` is marked inactive.
        iteration : int
            Global training iteration.
        """
        # ------------------------------------------------------------------------ #
        self.active = {name: False for name in self.names}

        for step in steps:
            value = sum(evaluation.objective.item() for evaluation in step.evaluations)

            # First value seen for this strategy becomes the baseline
            self.initial.setdefault(step.name, value)

            self.current[step.name] = value
            self.active[step.name] = True
            if self.path is not None:
                initial = self.initial[step.name]
                row = {"iteration": iteration, "ratio": value / initial if initial else 0.0,
                       "objective": value}
                for evaluation in step.evaluations:
                    for name, loss in evaluation.losses.items():
                        row[f"{evaluation.name}.{name}"] = loss
                self._pending.setdefault(step.name, []).append(row)

        # ------------------------------------------------------------------------ #
        return

    def display_if_due(self, iteration: int, final: bool) -> None:
        """Display progress at the configured interval and training endpoints."""
        if self.verbose and (iteration == 1 or iteration % self.interval == 0 or final):
            self.display(iteration)

    def render_live(self, iteration: int) -> str:
        """
        Build a live, multi-line summary meant to be redrawn every iteration.

        Parameters
        ----------
        iteration : int
            Current training iteration.

        Returns
        -------
        str
            One header line with the current iteration and training
            throughput, followed by one line per strategy with its
            objective function value and the ratio to its initial value.
            Strategies with no ``Step`` at the current iteration are
            tagged as inactive.
        """
        # ------------------------------------------------------------------------ #
        # Header
        rate = iteration / (time.perf_counter() - self._start_time + 1e-12)
        lines = [f"Iteration {iteration} - {rate:.1f} it/s"]
        
        # ------------------------------------------------------------------------ #
        # One line per strategy, in a fixed, known order
        width = max((len(name) for name in self.names), default=0)

        for name in self.names:
            current = self.current.get(name)
            initial = self.initial.get(name)

            if current is None:
                value = f"{'--':>9}"
                ratio = f"{'--':>9}"
            else:
                value = f"{current:>1.3e}"
                ratio = f"{(current / initial):>1.3e}"

            status = "[active]" if self.active[name] else "[inactive]"
            lines.append(f"- {name:<{width}}: func = {value}; ratio = {ratio} {status}")

        # ------------------------------------------------------------------------ #
        return "\n".join(lines)

    def display(self, iteration: int) -> None:
        """
        Print the live block, redrawing it in place instead of scrolling.

        Jupyter's output area does not interpret ANSI cursor-movement
        codes, so a different redraw strategy is used depending on where
        the code is running: ``IPython.display.clear_output`` inside a
        notebook kernel, and an ANSI cursor-up escape code in a regular
        terminal.

        Parameters
        ----------
        iteration : int
            Current training iteration.
        """
        # ------------------------------------------------------------------------ #
        block = self.render_live(iteration)

        from IPython import get_ipython

        if get_ipython() is not None:
            from IPython.display import clear_output

            clear_output(wait=True)
            print(block)

        else:
            num_lines = block.count("\n") + 1
            cursor_up = f"\x1b[{num_lines}A" if self._rendered else ""
            self._rendered = True

            print(cursor_up + "\r" + block.replace("\n", "\n\r"), flush=True)

        # ------------------------------------------------------------------------ #
        return

    def update_validation(self, iteration: int, losses: Dict[str, float]) -> None:
        """Record validation losses for the next save."""
        if self.path is not None:
            if "validation" in self.names:
                raise ValueError("The strategy name 'validation' conflicts with the validation history.")
            self._pending.setdefault("validation", []).append({"iteration": iteration, **losses})

    def record_times(self, iteration: int, optimization_s: float, elapsed_s: float) -> None:
        """Add cumulative optimization and wall time to active strategy histories."""
        for name in self.names:
            rows = self._pending.get(name)
            if rows and rows[-1]["iteration"] == iteration:
                rows[-1].update(optimization_s=optimization_s, elapsed_s=elapsed_s)

    def save(self) -> None:
        """Append pending rows once, preserving previously saved iterations."""
        if self.path is None or not self._pending:
            return

        self.path.mkdir(parents=True, exist_ok=True)
        for name in list(self._pending):
            rows = self._pending[name]
            header = list(rows[0])
            destination = self.path / f"{name}_{self.filename}"
            # Replace each CSV only after its previous and new rows are ready.
            with TemporaryDirectory(dir=self.path) as directory:
                temporary = Path(directory) / "history.csv"
                with temporary.open("w", newline="") as output:
                    writer = csv.writer(output, lineterminator="\n")
                    if destination.exists() and destination.stat().st_size:
                        with destination.open(newline="") as previous:
                            if next(csv.reader(previous)) != header:
                                raise ValueError(f"Incompatible history columns in {destination}.")
                            previous.seek(0)
                            copyfileobj(previous, output)
                    else:
                        writer.writerow(header)
                    for row in rows:
                        if row.keys() != rows[0].keys():
                            raise ValueError(f"Incompatible history columns in {destination}.")
                        writer.writerow([row["iteration"], *(f"{row[key]:.10e}" for key in header[1:])])
                temporary.replace(destination)
            del self._pending[name]

# -------------------------------------------------------------------------------- #
