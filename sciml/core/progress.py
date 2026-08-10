# -------------------------------------------------------------------------------- #
import os
import time

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
    strategy to its own log file via :meth:`log`.

    Note
    ----
    A strategy that is disabled at a given iteration (see ``Strategy.enable``)
    does not produce a ``Step``. Its line keeps showing the last known
    values, tagged as inactive, rather than being treated as zero or
    removed from the display.
    """

    def __init__(self, strategies: Sequence[Strategy]) -> None:
        """
        Parameters
        ----------
        strategies : Sequence[Strategy]
            Strategies tracked by this progress report. Their order fixes
            the order of the printed lines.
        """
        # ------------------------------------------------------------------------ #
        # Store constructor arguments
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

    def update(self, steps: Sequence[Step]) -> None:
        """
        Update the tracked statistics with the steps of the current iteration.

        Parameters
        ----------
        steps : Sequence[Step]
            Steps returned by the strategies at the current iteration. A
            strategy that did not produce a ``Step`` is marked inactive.
        """
        # ------------------------------------------------------------------------ #
        self.active = {name: False for name in self.names}

        for step in steps:
            value = sum(evaluation.objective.item() for evaluation in step.evaluations)

            # First value seen for this strategy becomes the baseline
            self.initial.setdefault(step.name, value)

            self.current[step.name] = value
            self.active[step.name] = True

        # ------------------------------------------------------------------------ #
        return

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

    def log(self, path: str, iteration: int, steps: Sequence[Step]) -> None:
        """
        Append one row per active strategy to its own log file.

        Each strategy gets its own text file, named ``<strategy>.log``
        inside ``path``, with one row per iteration where that strategy
        was active. Columns are, in order: the iteration, the objective
        function value, the individual losses of every evaluation in the
        step (named ``<objective>.<loss>``) and, last, the ratio to the
        initial objective value. The first column is always the
        iteration, so the file can later be imported (e.g. with
        ``pandas.read_csv``) to inspect the training history of a single
        strategy.

        Parameters
        ----------
        path : str
            Path to the results folder where the log files are stored.
            Created if it does not exist yet.
        iteration : int
            Current training iteration.
        steps : Sequence[Step]
            Steps returned by the strategies at the current iteration. A
            strategy with no ``Step`` here is inactive and no row is
            appended to its log file.
        """
        # ------------------------------------------------------------------------ #
        os.makedirs(path, exist_ok=True)

        for step in steps:
            value = sum(evaluation.objective.item() for evaluation in step.evaluations)
            initial = self.initial.get(step.name, value)
            ratio = value / initial if initial else 0.0

            losses = {
                f"{loss_name}": loss_value
                for evaluation in step.evaluations
                for loss_name, loss_value in evaluation.losses.items()
            }

            # -------------------------------------------------------------------- #
            file_path = os.path.join(path, f"{step.name}_log.csv")
            is_new_file = not os.path.exists(file_path)

            with open(file_path, "a") as file:
                if is_new_file:
                    header = ["iteration", "ratio", "objective", *losses.keys()]
                    file.write(",".join(header) + "\n")

                row = [
                    str(iteration),
                    f"{ratio:.10e}",
                    f"{value:.10e}",
                    *(f"{loss_value:.10e}" for loss_value in losses.values()),
                ]
                file.write(",".join(row) + "\n")

        # ------------------------------------------------------------------------ #
        return

# -------------------------------------------------------------------------------- #
