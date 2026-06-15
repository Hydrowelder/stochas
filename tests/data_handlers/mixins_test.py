"""Tests for NumericMixin, via NamedValue."""

import numpy as np

from stochas import NamedValue
from stochas.named_value import ValueName


def test_int_float_bool_casts():
    """Ensure casting dunders proxy to the stored value."""
    nv = NamedValue[int](name=ValueName("x"), stored_value=5)

    assert int(nv) == 5
    assert float(nv) == 5.0
    assert bool(nv) is True

    zero = NamedValue[int](name=ValueName("z"), stored_value=0)
    assert bool(zero) is False


def test_arithmetic_operators():
    """Ensure arithmetic dunders and their reflected counterparts proxy to the stored value."""
    nv = NamedValue[int](name=ValueName("x"), stored_value=5)

    assert nv + 1 == 6
    assert 1 + nv == 6
    assert nv - 1 == 4
    assert 1 - nv == -4
    assert nv * 2 == 10
    assert 2 * nv == 10
    assert nv / 2 == 2.5
    assert 2 / nv == 0.4
    assert nv**2 == 25


def test_arithmetic_with_another_named_value():
    """Ensure _extract unwraps a NamedValue operand via its .value."""
    a = NamedValue[int](name=ValueName("a"), stored_value=2)
    b = NamedValue[int](name=ValueName("b"), stored_value=3)

    assert a + b == 5


def test_comparison_operators():
    """Ensure ordering comparisons proxy to the stored value."""
    nv = NamedValue[int](name=ValueName("x"), stored_value=5)

    assert nv < 10
    assert nv <= 5
    assert nv > 1
    assert nv >= 5
    assert not (nv < 5)
    assert not (nv > 5)


def test_matmul():
    """Ensure __matmul__ proxies to the stored value's matrix multiplication."""
    nv = NamedValue[np.ndarray](name=ValueName("m"), stored_value=np.array([1, 2]))

    result = nv @ np.array([3, 4])

    assert result == 11


def test_len_getitem_and_contains():
    """Ensure sequence protocol dunders proxy to the stored value."""
    nv = NamedValue[np.ndarray](name=ValueName("a"), stored_value=np.array([1, 2, 3]))

    assert len(nv) == 3
    assert nv[0] == 1
    assert 2 in nv
    assert 5 not in nv


def test_array_conversion():
    """Ensure np.array() proxies to the stored value via __array__."""
    nv = NamedValue[np.ndarray](name=ValueName("a"), stored_value=np.array([1, 2, 3]))

    arr = np.array(nv)

    assert isinstance(arr, np.ndarray)
    np.testing.assert_array_equal(arr, [1, 2, 3])


def test_shape_ndim_dtype_size():
    """Ensure array-like properties proxy to the stored value."""
    nv = NamedValue[np.ndarray](
        name=ValueName("a"), stored_value=np.array([[1, 2], [3, 4]])
    )

    assert nv.shape == (2, 2)
    assert nv.ndim == 2
    assert nv.size == 4
    assert nv.dtype == np.array([[1, 2], [3, 4]]).dtype


def test_squeeze_updates_stored_value_in_place():
    """Ensure squeeze removes size-one axes and writes the result back via force_set_value."""
    nv = NamedValue[np.ndarray](name=ValueName("a"), stored_value=np.array([[1], [2]]))

    result = nv.squeeze()

    assert result is nv
    np.testing.assert_array_equal(nv.value, [1, 2])
    assert nv.value.shape == (2,)
