"""Tests for NamedValue."""

import pickle

import pytest
from pydantic import ValidationError

from stochas import NamedValue, NamedValueState
from stochas.named_value import UNSET_SENTINEL, ValueName


def test_initial_state_unset():
    nv = NamedValue[int](name=ValueName("x"))

    assert nv.state is NamedValueState.UNSET
    assert nv.is_set is False

    with pytest.raises(ValueError, match="has not been set"):
        _ = nv.value


def test_set_value_once():
    nv = NamedValue[int](name=ValueName("x"))

    nv.value = 10

    assert nv.state is NamedValueState.SET
    assert nv.is_set is True
    assert nv.value == 10


def test_double_set_raises():
    nv = NamedValue[int](name=ValueName("x"))
    nv.value = 10

    with pytest.raises(ValueError, match="already been set and is frozen"):
        nv.value = 20


def test_force_set_overwrites():
    nv = NamedValue[int](name=ValueName("x"))
    nv.value = 10

    nv.force_set_value(20)

    assert nv.value == 20
    assert nv.state is NamedValueState.SET


def test_type_enforcement_on_assignment():
    nv = NamedValue[int](name=ValueName("x"))

    with pytest.raises(ValidationError):
        nv.value = "not an int"  # type: ignore[arg-type]


def test_validate_assignment_enforced():
    nv = NamedValue[int](name=ValueName("x"))
    nv.value = 5

    with pytest.raises(ValidationError):
        nv.stored_value = "bad"  # type: ignore[assignment]


def test_serialization_round_trip():
    nv = NamedValue[int](name=ValueName("x"))
    nv.value = 42

    dumped = nv.model_dump_json()
    loaded = NamedValue[int].model_validate_json(dumped)

    assert loaded.name == "x"
    assert loaded.value == 42
    assert loaded.state is NamedValueState.SET


def test_unset_serialization_round_trip():
    nv = NamedValue[int](name=ValueName("x"))

    dumped = nv.model_dump_json()
    loaded = NamedValue[int].model_validate_json(dumped)

    assert loaded.state is NamedValueState.UNSET
    assert loaded.is_set is False

    with pytest.raises(ValueError):
        _ = loaded.value


def test_extra_fields_forbidden():
    with pytest.raises(ValidationError):
        NamedValue[int](name=ValueName("x"), extra_field=123)  # type: ignore[arg-type]


def test_string_named_value():
    nv = NamedValue[str](name=ValueName("username"))
    nv.value = "david"

    assert isinstance(nv.value, str)
    assert nv.value == "david"


def test_pickle_roundtrip_preserves_generic_type():
    """Ensure __reduce__ preserves the generic type parameter when pickling and unpickling."""
    nv = NamedValue[int](name=ValueName("count"), stored_value=7)

    restored = pickle.loads(pickle.dumps(nv))

    assert restored == nv
    assert restored.__class__ is nv.__class__


def test_validate_state_set_without_stored_value_raises():
    """Ensure declaring state=SET without a stored_value is rejected."""
    with pytest.raises(ValueError, match=r"cannot be set to `NamedValueState\.UNSET`"):
        NamedValue[int](name=ValueName("x"), state=NamedValueState.SET)


def test_value_getter_raises_runtime_error_on_corrupted_state():
    """Ensure the value getter detects a SET state with a sentinel stored_value."""
    nv = NamedValue[int].model_construct(
        name=ValueName("x"), state=NamedValueState.SET, stored_value=UNSET_SENTINEL
    )

    with pytest.raises(RuntimeError, match="implying something was corrupted"):
        _ = nv.value


def test_value_getter_raises_not_implemented_for_unknown_state():
    """Ensure the value getter rejects an unrecognized state value."""
    nv = NamedValue[int].model_construct(
        name=ValueName("x"), state="bogus", stored_value=1
    )

    with pytest.raises(NotImplementedError, match="has not been implemented"):
        _ = nv.value


def test_value_setter_raises_not_implemented_for_unknown_state():
    """Ensure the value setter rejects an unrecognized state value."""
    nv = NamedValue[int].model_construct(
        name=ValueName("x"), state="bogus", stored_value=1
    )

    with pytest.raises(NotImplementedError, match="has not been implemented"):
        nv.value = 5


def test_generic_type_propagation_for_type_checkers():
    nv = NamedValue[int](name=ValueName("count"))
    nv.value = 7

    def takes_int(x: int) -> None:
        pass

    # This should type-check statically
    takes_int(nv.value)
