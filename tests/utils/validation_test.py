import pytest
import torch

from sciml.metrics.mean_absolute_error import MeanAbsoluteError
from sciml.utils import validation


@pytest.mark.parametrize(
    "validator,valid,invalid",
    [
        (validation.is_integer, 1, 1.0),
        (validation.is_float, 1.0, 1),
        (validation.is_string, "x", 1),
        (validation.is_callable, lambda x: x, 1),
    ],
)
def test_basic_validators_accept_valid_values_and_reject_invalid_values(validator, valid, invalid):
    assert validator(valid) is None
    with pytest.raises(TypeError):
        validator(invalid)


def test_boolean_validator_is_available_at_module_level():
    assert validation.is_boolean(True) is None
    with pytest.raises(TypeError):
        validation.is_boolean(1)


def test_iterable_and_mapping_validators_check_content():
    assert validation.is_iterable([1, 2], dtype=int) is None
    assert validation.is_mapping({"x": torch.tensor([1.0])}, keys=["x"], dtype=torch.Tensor) is None

    with pytest.raises(TypeError):
        validation.is_iterable([1, "2"], dtype=int)
    with pytest.raises(KeyError):
        validation.is_mapping({}, keys=["x"])


def test_object_validators_accept_expected_types_and_reject_invalid_types():
    network = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(network.parameters(), lr=0.1)

    assert validation.is_metric(MeanAbsoluteError()) is None
    assert validation.is_network(network) is None
    assert validation.is_optimizer(optimizer) is None

    with pytest.raises(TypeError):
        validation.is_metric(object())
    with pytest.raises(TypeError):
        validation.is_network(object())
    with pytest.raises(TypeError):
        validation.is_optimizer(object())
