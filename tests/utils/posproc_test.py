import math

import pandas as pd
import pytest

from sciml.utils.posproc import plot_history, summarize


def test_plot_history_saves_figure(tmp_path) -> None:
    df = pd.DataFrame(
        {
            "iteration": [0, 1, 2],
            "loss": [1.0, 0.5, 0.25],
            "metric": [0.8, 0.6, 0.4],
        }
    )
    save_path = tmp_path / "history.png"

    plot_history(
        df=df,
        y_keys=["loss", "metric"],
        x_key="iteration",
        labels={"loss": "Loss", "metric": "Metric"},
        title="Optimization history",
        ylabel="Value",
        save_path=str(save_path),
        show=False,
    )

    assert save_path.is_file()


def test_plot_history_raises_error_for_invalid_x_key() -> None:
    df = pd.DataFrame({"loss": [1.0, 0.5, 0.25]})

    with pytest.raises(KeyError, match="Column 'iteration' not found"):
        plot_history(
            df=df,
            y_keys=["loss"],
            x_key="iteration",
            show=False,
        )


def test_summarize_min_mode() -> None:
    df = pd.DataFrame(
        {
            "iteration": [0, 1, 2, 3],
            "loss": [10.0, 5.0, 2.0, 4.0],
        }
    )

    result = summarize(
        df=df,
        keys=["loss"],
        iteration_key="iteration",
        last_n=2,
        mode="min",
    )

    row = result.iloc[0]

    assert row["metric"] == "loss"
    assert row["initial"] == 10.0
    assert row["final"] == 4.0
    assert row["best"] == 2.0
    assert row["best_iteration"] == 2
    assert row["mean_last_2"] == 3.0
    assert row["std_last_2"] == pytest.approx(math.sqrt(2.0))
    assert row["improvement_abs"] == 6.0
    assert row["improvement_rel"] == pytest.approx(0.6)


def test_summarize_max_mode() -> None:
    df = pd.DataFrame(
        {
            "iteration": [0, 1, 2, 3],
            "score": [0.10, 0.30, 0.25, 0.20],
        }
    )

    result = summarize(
        df=df,
        keys=["score"],
        iteration_key="iteration",
        last_n=2,
        mode="max",
    )

    row = result.iloc[0]

    assert row["metric"] == "score"
    assert row["initial"] == 0.10
    assert row["final"] == 0.20
    assert row["best"] == 0.30
    assert row["best_iteration"] == 1
    assert row["mean_last_2"] == pytest.approx(0.225)


def test_summarize_uses_numeric_columns_when_keys_are_none() -> None:
    df = pd.DataFrame(
        {
            "iteration": [0, 1, 2],
            "loss": [3.0, 2.0, 1.0],
            "mae": [1.0, 0.8, 0.5],
            "label": ["a", "b", "c"],
        }
    )

    result = summarize(
        df=df,
        keys=None,
        iteration_key="iteration",
        last_n=2,
    )

    assert result["metric"].tolist() == ["loss", "mae"]


def test_summarize_raises_error_for_invalid_mode() -> None:
    df = pd.DataFrame({"loss": [1.0, 0.5, 0.25]})

    with pytest.raises(ValueError, match="mode must be either 'min' or 'max'"):
        summarize(df=df, keys=["loss"], mode="invalid")
