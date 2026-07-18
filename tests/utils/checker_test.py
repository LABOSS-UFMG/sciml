import pytest
from collections.abc import Iterable, Mapping

from sciml.utils.checker import (
    is_integer,
    is_float,
    is_string,
    is_boolean,
    is_callable,
    is_iterable,
    is_dtype,
)


def test_is_integer():
    """Test integer validation."""
    # Should not raise
    is_integer(1)
    is_integer(0)
    is_integer(-5)
    
    # Should raise TypeError
    with pytest.raises(TypeError, match="Input must be of type int"):
        is_integer(1.5)
    with pytest.raises(TypeError, match="Input must be of type int"):
        is_integer("1")
    with pytest.raises(TypeError, match="Input must be of type int"):
        is_integer(True)


def test_is_float():
    """Test float validation."""
    # Should not raise
    is_float(1.5)
    is_float(0.0)
    is_float(-3.14)
    
    # Should raise TypeError
    with pytest.raises(TypeError, match="Input must be of type float"):
        is_float(1)
    with pytest.raises(TypeError, match="Input must be of type float"):
        is_float("1.5")
    with pytest.raises(TypeError, match="Input must be of type float"):
        is_float(True)


def test_is_string():
    """Test string validation."""
    # Should not raise
    is_string("hello")
    is_string("")
    is_string("123")
    
    # Should raise TypeError
    with pytest.raises(TypeError, match="Input must be of type str"):
        is_string(123)
    with pytest.raises(TypeError, match="Input must be of type str"):
        is_string(1.5)
    with pytest.raises(TypeError, match="Input must be of type str"):
        is_string(True)


def test_is_boolean():
    """Test boolean validation."""
    # Should not raise
    is_boolean(True)
    is_boolean(False)
    
    # Should raise TypeError
    with pytest.raises(TypeError, match="Input must be of type bool"):
        is_boolean(1)
    with pytest.raises(TypeError, match="Input must be of type bool"):
        is_boolean("True")
    with pytest.raises(TypeError, match="Input must be of type bool"):
        is_boolean(None)


def test_is_callable():
    """Test callable validation."""
    # Should not raise
    def func(): pass
    is_callable(func)
    is_callable(lambda x: x)
    is_callable(print)
    
    # Should raise TypeError
    with pytest.raises(TypeError, match="Input must be callable"):
        is_callable(123)
    with pytest.raises(TypeError, match="Input must be callable"):
        is_callable("hello")
    with pytest.raises(TypeError, match="Input must be callable"):
        is_callable([1, 2, 3])


def test_is_iterable():
    """Test iterable validation with and without dtype constraints."""
    # Should not raise - basic iterable
    is_iterable([1, 2, 3])
    is_iterable((1, 2, 3))
    is_iterable("hello")
    is_iterable(range(5))
    is_iterable({1, 2, 3})
    
    # Should not raise - with dtype constraint
    is_iterable([1, 2, 3], dtype=int)
    is_iterable((1.5, 2.5), dtype=float)
    is_iterable(["a", "b"], dtype=str)
    is_iterable([True, False], dtype=bool)
    
    # Should raise TypeError - not iterable
    with pytest.raises(TypeError, match="Input must be of type Iterable"):
        is_iterable(123)
    with pytest.raises(TypeError, match="Input must be of type Iterable"):
        is_iterable(None)
    
    # Should raise TypeError - dtype mismatch
    with pytest.raises(TypeError, match="Iterable element at index 1 must be of type <class 'int'>"):
        is_iterable([1, "2", 3], dtype=int)
    with pytest.raises(TypeError, match="Iterable element at index 1 must be of type <class 'str'>"):
        is_iterable(["a", 1, "b"], dtype=str)


def test_is_dtype():
    """Test generic dtype validation."""
    # Should not raise
    is_dtype(123, int)
    is_dtype(1.5, float)
    is_dtype("hello", str)
    is_dtype(True, bool)
    is_dtype([1, 2, 3], list)
    is_dtype({"a": 1}, dict)
    
    # Should raise TypeError
    with pytest.raises(TypeError, match="Input must be an instance of <class 'int'>"):
        is_dtype(1.5, int)
    with pytest.raises(TypeError, match="Input must be an instance of <class 'str'>"):
        is_dtype(123, str)
    with pytest.raises(TypeError, match="Input must be an instance of <class 'float'>"):
        is_dtype(1, float)