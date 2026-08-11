# -------------------------------------------------------------------------------- #
import pandas as pd
import matplotlib.pyplot as plt

from typing import Sequence, Optional, Tuple

# -------------------------------------------------------------------------------- #
def plot_history(
        csv: str | pd.DataFrame,
        y_keys: Sequence[str],
        x_key: str = "iteration",
        figsize: Tuple[float] = (6, 4),
        legend: Optional[Sequence[str]] = None,
        y_label: Optional[str] = None,
        x_label: Optional[str] = "Iteration",
        yscale: str = "log",
        filename: Optional[str] = None,
        extension: str = "png",
        ) -> None:
    """
    Plot the optimization history in a csv file.

    Parameters
    ----------
    csv : str
        Path to the csv file containing the optimization history.
    y_keys : Sequence[str]
        List of keys to plot on the y-axis.
    x_key : str, optional
        Key to plot on the x-axis. Default is "iteration".
    legend : Optional[Sequence[str]], optional
        List of legend labels for the y_keys. If None, the y_keys will be used as
        legend labels. Default is None.
    y_label : Optional[str], optional
        Label for the y-axis. If None, the y_keys will be used as the y-axis label.
        Default is None.
    x_label : Optional[str], optional
        Label for the x-axis. If None, the x_key will be used as the x-axis label.
        Default is None.
    yscale : str, optional
        Scale of the y-axis. Default is "log". Other options are "linear", "symlog",
        "logit", "function", "functionlog", "functionsymlog
    """
    # ------------------------------------------------------------------------ #
    # Update the function to read the csv file and plot the optimization history
    if isinstance(csv, str):
        data = pd.read_csv(csv)
    else:
        data = csv

    # ------------------------------------------------------------------------ #
    # Plot the optimization history
    plt.figure(figsize=figsize)

    for i, y_key in enumerate(y_keys):
        plt.plot(data[x_key], data[y_key], label=legend[i] if legend is not None else y_key)

    if x_label is not None: plt.xlabel(x_key)
    if y_label is not None: plt.ylabel(y_label)
    plt.legend(legend)
    plt.grid(True)

    plt.yscale(yscale)
    plt.tight_layout()

    if filename:
        plt.savefig(filename, format=extension)

    plt.show()

    # ------------------------------------------------------------------------ #
    return

# -------------------------------------------------------------------------------- #
