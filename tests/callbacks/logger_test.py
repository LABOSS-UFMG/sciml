import pytest
import torch
import tempfile
import os
import pandas as pd
import warnings

from sciml.callbacks.logger import Logger
from sciml.core.evaluation import Evaluation


# Suppress CUDA warnings
@pytest.fixture(autouse=True)
def suppress_cuda_warnings():
    """Suppress CUDA compatibility warnings."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module="torch.cuda")
        yield


@pytest.fixture(autouse=True)
def force_cpu():
    """Force all tensors to be created on CPU to avoid CUDA warnings."""
    original_device = torch.get_default_device()
    torch.set_default_device('cpu')
    yield
    torch.set_default_device(original_device)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sample_losses():
    """Create sample loss evaluations."""
    loss1 = Evaluation(
        values={"mse": torch.tensor(0.5), "l1": torch.tensor(0.3)},
        weights={"mse": 1.0, "l1": 0.5},
        metadata={"name": "training_loss"}
    )
    loss2 = Evaluation(
        values={"cross_entropy": torch.tensor(1.2)},
        weights={"cross_entropy": 1.0},
        metadata={"name": "classification_loss"}
    )
    return [loss1, loss2]


@pytest.fixture
def sample_metrics():
    """Create sample metric evaluations."""
    metric1 = Evaluation(
        values={"accuracy": torch.tensor(0.95), "precision": torch.tensor(0.92)},
        weights={},
        metadata={"name": "validation_metrics"}
    )
    metric2 = Evaluation(
        values={"loss": torch.tensor(0.45)},
        weights={},
        metadata={"name": "test_metrics"}
    )
    return [metric1, metric2]


def test_logger_init(temp_dir):
    """Test Logger initialization."""
    logger = Logger(directory=temp_dir)
    
    assert logger.directory == temp_dir.rstrip("/")
    assert os.path.exists(logger.directory)


def test_logger_init_creates_directory(temp_dir):
    """Test that Logger creates the directory if it doesn't exist."""
    nested_dir = os.path.join(temp_dir, "nested", "logs")
    logger = Logger(directory=nested_dir)
    
    assert os.path.exists(nested_dir)
    assert logger.directory == nested_dir


def test_logger_on_iteration_end_first_iteration(temp_dir, sample_losses):
    """Test logging at first iteration (creates header)."""
    logger = Logger(directory=temp_dir)
    
    # Log at iteration 1
    logger.on_iteration_end(iteration=1, losses=sample_losses, metrics=None)
    
    # Check files were created
    filename = os.path.join(temp_dir, "training_loss.csv")
    assert os.path.exists(filename)
    
    # Read the file and check content
    df = pd.read_csv(filename)
    assert len(df) == 1
    assert df.columns.tolist() == ["iteration", "Objective function", "mse", "l1"]
    assert df.iloc[0]["iteration"] == 1
    # Objective function = sum of weighted losses: 0.5*1.0 + 0.3*0.5 = 0.65
    # Use approx for floating point comparison
    assert df.iloc[0]["Objective function"] == pytest.approx(0.65)
    assert df.iloc[0]["mse"] == pytest.approx(0.5)
    assert df.iloc[0]["l1"] == pytest.approx(0.3)


def test_logger_on_iteration_end_multiple_iterations(temp_dir, sample_losses):
    """Test logging at multiple iterations."""
    logger = Logger(directory=temp_dir)
    
    # Log at iterations 1, 2, 3
    for i in range(1, 4):
        logger.on_iteration_end(iteration=i, losses=sample_losses, metrics=None)
    
    # Check file content
    filename = os.path.join(temp_dir, "training_loss.csv")
    df = pd.read_csv(filename)
    assert len(df) == 3
    assert df["iteration"].tolist() == [1, 2, 3]


def test_logger_on_iteration_end_with_metrics(temp_dir, sample_losses, sample_metrics):
    """Test logging with metrics."""
    logger = Logger(directory=temp_dir)
    
    # Log at iteration 1 with metrics
    logger.on_iteration_end(iteration=1, losses=sample_losses, metrics=sample_metrics)
    
    # Check loss file
    loss_file = os.path.join(temp_dir, "training_loss.csv")
    assert os.path.exists(loss_file)
    df_loss = pd.read_csv(loss_file)
    assert df_loss.columns.tolist() == ["iteration", "Objective function", "mse", "l1"]
    
    # Check metric file
    metric_file = os.path.join(temp_dir, "validation_metrics.csv")
    assert os.path.exists(metric_file)
    df_metric = pd.read_csv(metric_file)
    assert df_metric.columns.tolist() == ["iteration", "accuracy", "precision"]
    # Use approx for floating point comparisons
    assert df_metric.iloc[0]["accuracy"] == pytest.approx(0.95)
    assert df_metric.iloc[0]["precision"] == pytest.approx(0.92)
    
    # Check second metric file
    metric_file2 = os.path.join(temp_dir, "test_metrics.csv")
    assert os.path.exists(metric_file2)
    df_metric2 = pd.read_csv(metric_file2)
    assert df_metric2.columns.tolist() == ["iteration", "loss"]
    assert df_metric2.iloc[0]["loss"] == pytest.approx(0.45)


def test_logger_on_iteration_end_multiple_losses(temp_dir):
    """Test logging with multiple loss functions."""
    logger = Logger(directory=temp_dir)
    
    loss1 = Evaluation(
        values={"mse": torch.tensor(0.5)},
        weights={"mse": 1.0},
        metadata={"name": "loss_a"}
    )
    loss2 = Evaluation(
        values={"l1": torch.tensor(0.3)},
        weights={"l1": 0.5},
        metadata={"name": "loss_b"}
    )
    
    logger.on_iteration_end(iteration=1, losses=[loss1, loss2], metrics=None)
    
    # Check first loss file
    file_a = os.path.join(temp_dir, "loss_a.csv")
    assert os.path.exists(file_a)
    df_a = pd.read_csv(file_a)
    assert df_a.columns.tolist() == ["iteration", "Objective function", "mse"]
    assert df_a.iloc[0]["mse"] == pytest.approx(0.5)
    
    # Check second loss file
    file_b = os.path.join(temp_dir, "loss_b.csv")
    assert os.path.exists(file_b)
    df_b = pd.read_csv(file_b)
    assert df_b.columns.tolist() == ["iteration", "Objective function", "l1"]
    assert df_b.iloc[0]["l1"] == pytest.approx(0.3)


def test_logger_to_dataframe(temp_dir, sample_losses):
    """Test loading logged data into a DataFrame."""
    logger = Logger(directory=temp_dir)
    
    # Log some data
    for i in range(1, 4):
        logger.on_iteration_end(iteration=i, losses=sample_losses, metrics=None)
    
    # Load data into DataFrame
    df = logger.to_dataframe("training_loss")
    
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3
    assert df.columns.tolist() == ["iteration", "Objective function", "mse", "l1"]
    assert df["iteration"].tolist() == [1, 2, 3]


def test_logger_to_dataframe_with_metrics(temp_dir, sample_losses, sample_metrics):
    """Test loading metric data into a DataFrame."""
    logger = Logger(directory=temp_dir)
    
    # Log with metrics
    for i in range(1, 4):
        logger.on_iteration_end(iteration=i, losses=sample_losses, metrics=sample_metrics)
    
    # Load metric data
    df = logger.to_dataframe("validation_metrics")
    
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3
    assert df.columns.tolist() == ["iteration", "accuracy", "precision"]
    assert df["iteration"].tolist() == [1, 2, 3]


def test_logger_to_dataframe_file_not_found(temp_dir):
    """Test to_dataframe with non-existent file."""
    logger = Logger(directory=temp_dir)
    
    with pytest.raises(FileNotFoundError):
        logger.to_dataframe("non_existent")


def test_logger_no_op_methods(temp_dir):
    """Test that no-op methods don't raise errors."""
    logger = Logger(directory=temp_dir)
    
    # These should not raise any errors
    logger.on_train_start()
    logger.on_iteration_start()
    logger.on_train_end()
    logger.on_exception()
    
    # No files should have been created
    assert len(os.listdir(temp_dir)) == 0


def test_logger_appends_to_existing_files(temp_dir, sample_losses):
    """Test that Logger appends to existing files rather than overwriting."""
    logger = Logger(directory=temp_dir)
    
    # First logging
    logger.on_iteration_end(iteration=1, losses=sample_losses, metrics=None)
    
    # Second logging
    logger.on_iteration_end(iteration=2, losses=sample_losses, metrics=None)
    
    # Check file has both rows
    filename = os.path.join(temp_dir, "training_loss.csv")
    df = pd.read_csv(filename)
    assert len(df) == 2
    assert df["iteration"].tolist() == [1, 2]


def test_logger_handles_tensors_properly(temp_dir):
    """Test that Logger correctly handles tensor values."""
    logger = Logger(directory=temp_dir)
    
    # Create evaluation with different tensor types
    loss = Evaluation(
        values={
            "loss1": torch.tensor(0.123456),
            "loss2": torch.tensor(0.789012)
        },
        weights={"loss1": 1.0, "loss2": 1.0},
        metadata={"name": "test_loss"}
    )
    
    logger.on_iteration_end(iteration=1, losses=[loss], metrics=None)
    
    # Check values are correctly converted to Python floats
    filename = os.path.join(temp_dir, "test_loss.csv")
    df = pd.read_csv(filename)
    # Use approx for floating point comparisons
    assert df.iloc[0]["loss1"] == pytest.approx(0.123456)
    assert df.iloc[0]["loss2"] == pytest.approx(0.789012)
    assert df.iloc[0]["Objective function"] == pytest.approx(0.123456 + 0.789012)