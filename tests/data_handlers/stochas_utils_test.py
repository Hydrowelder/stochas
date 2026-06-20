"""Tests for the pydantic-generic pickling helpers in stochas.utils."""

import pickle

from stochas import NamedValue, NamedValueDict, NamedValueList
from stochas.named_value import ValueName
from stochas.utils import _describe_type, _resolve_type


def test_describe_and_resolve_plain_type():
    """Ensure non-generic types pass through _describe_type and _resolve_type unchanged."""
    assert _describe_type(int) is int
    assert _resolve_type(int) is int


def test_describe_and_resolve_nested_generic_type():
    """Ensure a generic type with generic args is described and resolved recursively."""
    tp = NamedValueDict[NamedValue[int]]

    description = _describe_type(tp)
    resolved = _resolve_type(description)

    assert resolved is tp


def test_pickle_roundtrip_for_nested_generic():
    """Ensure pickling a doubly-parameterized generic preserves its full type."""
    d = NamedValueDict[NamedValue[int]]()
    d.update(NamedValue[NamedValue[int]](name=ValueName("x")))

    restored = pickle.loads(pickle.dumps(d))

    assert restored == d
    assert restored.__class__ is d.__class__


def test_pickle_roundtrip_for_unparameterized_generic():
    """Ensure pickling an unparameterized generic model preserves its (origin) class."""
    d = NamedValueDict()
    d.update(NamedValue[int](name=ValueName("x"), stored_value=1))

    restored = pickle.loads(pickle.dumps(d))

    assert restored == d
    assert restored.__class__ is d.__class__


def test_pickle_roundtrip_for_unparameterized_list():
    """Ensure pickling an unparameterized generic list preserves its (origin) class."""
    nv_list = NamedValueList()
    nv_list.append(NamedValue[int](name=ValueName("x"), stored_value=1))

    restored = pickle.loads(pickle.dumps(nv_list))

    assert restored == nv_list
    assert restored.__class__ is nv_list.__class__
