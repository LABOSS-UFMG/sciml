#####################################################################################
from typing import Iterable, Optional

import pandas as pd
import matplotlib.pyplot as plt

#####################################################################################
def plot_history(
        df: pd.DataFrame,
        y_keys: Iterable[str],
        x_key: Optional[str] = None,
        labels: Optional[dict[str, str]] = None,
        title: Optional[str] = None,
        xlabel: str = "Iteration",
        ylabel: Optional[str] = None,
        figsize: tuple[float, float] = (10, 6),
        linewidth: float = 2.0,
        linestyle: str = "-",
        font_size: float = 14,
        marker: Optional[str] = None,
        grid: bool = True,
        yscale: str = "log",
        xscale: str = "linear",
        save_path: Optional[str] = None,
        show: bool = True,
    ) -> None:
    """
    Plot losses or metrics from a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the values to plot.
    y_keys : Iterable[str]
        Column names to plot.
    x_key : str | None, optional
        Column name used as x-axis. If None, the DataFrame index is used.
    title : str, optional
        Figure title.
    xlabel : str, optional
        Label of the x-axis.
    ylabel : str, optional
        Label of the y-axis.
    labels : dict[str, str] | None, optional
        Custom labels for each plotted key.
        Example: {"train_loss": "Train Loss"}.
    figsize : tuple[float, float], optional
        Figure size.
    linewidth : float, optional
        Line width.
    linestyle : str, optional
        Line style.
    font_size : float
        Font size for labels and title.
    marker : str | None, optional
        Marker style.
    grid : bool, optional
        Whether to show the grid.
    yscale : str, optional
        Scale of the y-axis. Common values are "linear" and "log".
    xscale : str, optional
        Scale of the x-axis. Common values are "linear" and "log".
    save_path : str | Path | None, optional
        Path to save the figure. If None, the figure is not saved.
    show : bool, optional
        Whether to display the figure.
    """
    # ---------------------------------------------------------------------------
    legend = False if labels is None else True

    # ---------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=figsize)

    # ---------------------------------------------------------------------------
    if x_key is not None:
        if x_key not in df.columns:
            raise KeyError(f"Column '{x_key}' not found in DataFrame.")
        x = df[x_key]
    else:
        x = df.index

    # ---------------------------------------------------------------------------
    for key in y_keys:
        if key not in df.columns:
            print(f"Warning: column '{key}' not found in DataFrame.")
            continue

        label = labels.get(key, key) if labels is not None else key

        if "$" in label:
            label = rf"{label}"
        
        ax.plot(
            x,
            df[key],
            label=label,
            linewidth=linewidth,
            linestyle=linestyle,
            marker=marker,
        )

    # ---------------------------------------------------------------------------
    ax.set_title(title, fontsize=font_size)
    ax.set_xlabel(xlabel, fontsize=font_size)
    ax.set_ylabel(ylabel, fontsize=font_size)
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)

    # ---------------------------------------------------------------------------
    if grid:
        ax.grid(True)

    if legend:
        ax.legend(fontsize=font_size)

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    
    # ---------------------------------------------------------------------------
    return


def summarize(
        df: pd.DataFrame,
        keys: Optional[Iterable[str]] = None,
        iteration_key: Optional[str] = None,
        last_n: int = 100,
        mode: str = "min",
    ) -> pd.DataFrame:
    """
    Summarize optimization history.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing optimization history.
    keys : Iterable[str] | None, optional
        Column names to summarize. If None, all numeric columns are used.
    iteration_key : str | None, optional
        Column name containing the iteration values. If None, the DataFrame index is used.
    last_n : int, optional
        Number of last iterations used to compute final statistics.
    mode : str, optional
        Optimization mode. Use "min" for losses/errors and "max" for scores.

    Returns
    -------
    pd.DataFrame
        DataFrame containing summary statistics for each selected key.
    """

    # ---------------------------------------------------------------------------
    if mode not in {"min", "max"}:
        raise ValueError("mode must be either 'min' or 'max'.")

    if iteration_key is not None and iteration_key not in df.columns:
        raise KeyError(f"Column '{iteration_key}' not found in DataFrame.")

    if keys is None:
        keys = df.select_dtypes(include="number").columns.tolist()

        if iteration_key is not None and iteration_key in keys:
            keys.remove(iteration_key)

    if iteration_key is not None:
        iterations = df[iteration_key]
    else:
        iterations = df.index.to_series(index=df.index)

    # ---------------------------------------------------------------------------
    summary = []

    for key in keys:
    
        # -----------------------------------------------------------------------
        if key not in df.columns:
            print(f"Warning: column '{key}' not found in DataFrame.")
            continue
        
        # -----------------------------------------------------------------------
        values = pd.to_numeric(df[key], errors="coerce").dropna()

        if values.empty:
            print(f"Warning: column '{key}' has no valid numeric values.")
            continue
        
        # -----------------------------------------------------------------------
        selected_iterations = iterations.loc[values.index]

        # -----------------------------------------------------------------------
        initial_value = values.iloc[0]
        final_value = values.iloc[-1]

        # -----------------------------------------------------------------------
        if mode == "min":
            best_index = values.idxmin()
        else:
            best_index = values.idxmax()

        # -----------------------------------------------------------------------
        best_value = values.loc[best_index]
        best_iteration = selected_iterations.loc[best_index]

        # -----------------------------------------------------------------------
        last_values = values.tail(last_n)

        mean_last = last_values.mean()
        std_last = last_values.std()

        # -----------------------------------------------------------------------
        improvement_abs = initial_value - final_value

        # -----------------------------------------------------------------------
        if initial_value != 0:
            improvement_rel = improvement_abs / abs(initial_value)
        else:
            improvement_rel = float("nan")

        # -----------------------------------------------------------------------
        summary.append(
            {
                "metric": key,
                "initial": initial_value,
                "final": final_value,
                "best": best_value,
                "best_iteration": best_iteration,
                f"mean_last_{last_n}": mean_last,
                f"std_last_{last_n}": std_last,
                "improvement_abs": improvement_abs,
                "improvement_rel": improvement_rel,
            }
        )

    # ---------------------------------------------------------------------------
    return pd.DataFrame(summary)

#####################################################################################
