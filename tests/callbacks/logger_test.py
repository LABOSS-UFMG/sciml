import pandas as pd
import pytest
import torch

from sciml.callbacks.logger import Logger


def test_logger_creates_history_file_and_records_rows(tmp_path):
    logger = Logger(
        directory=str(tmp_path),
        loss_names=["loss"],
        metric_names=["metric"],
        frequency=2,
    )

    logger.on_iteration_end(iteration=1, losses={"loss": 1.0}, metrics={"metric": 0.5})
    logger.on_iteration_end(iteration=2, losses={"loss": 2.0}, metrics={"metric": 0.25})

    history = logger.to_dataframe()

    assert list(history.columns) == ["iteration", "loss", "metric"]
    assert len(history) == 1
    assert history.loc[0, "iteration"] == 2
    assert history.loc[0, "loss"] == 2.0
    assert history.loc[0, "metric"] == 0.25


def test_logger_writes_network_and_optimizer_info(tmp_path):
    logger = Logger(directory=str(tmp_path), loss_names=["loss"])
    network = torch.nn.Linear(2, 1)
    optimizer = torch.optim.Adam(network.parameters(), lr=0.01)

    logger.on_train_start(network=network, optimizer=optimizer)

    info = (tmp_path / "info.txt").read_text()
    assert "num_trainable_params: 3" in info
    assert "learning_rate: 0.01" in info


def test_logger_noop_hooks_accept_empty_calls(tmp_path):
    logger = Logger(directory=str(tmp_path), loss_names=["loss"])

    assert logger.on_iteration_start() is None
    assert logger.on_train_end() is None


def test_logger_rejects_unexpected_loss_names(tmp_path):
    logger = Logger(directory=str(tmp_path), loss_names=["expected"])

    with pytest.raises(ValueError):
        logger.on_iteration_end(iteration=1, losses={"other": 1.0})


def test_logger_rejects_undeclared_metric_names(tmp_path):
    logger = Logger(directory=str(tmp_path), loss_names=["loss"], metric_names=["known"])

    with pytest.raises(ValueError):
        logger.on_iteration_end(iteration=1, losses={"loss": 1.0}, metrics={"unknown": 0.0})
