from __future__ import annotations

import logging
from enum import StrEnum
from typing import Annotated, Any, Literal, NewType, Self, cast

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    FieldSerializationInfo,
    field_serializer,
    model_validator,
)
from pydantic_core import to_jsonable_python

from stochas.base_collections import BaseDict, BaseList
from stochas.mixins import NumericMixin

__all__ = [
    "NamedValue",
    "NamedValueDict",
    "NamedValueList",
    "NamedValueState",
    "Val",
    "ValueName",
]

logger = logging.getLogger(__name__)

ValueName = NewType("ValueName", str)
"""Alias of string. Used to type hint a named value's name."""


UNSET_SENTINEL = "__STOCHAS_UNSET_SENTINEL__"
UnsetType = Literal["__STOCHAS_UNSET_SENTINEL__"]


class Val(BaseModel):
    """
    Defines a reference to a variable name.

    This acts as a 'pointer' or 'placeholder' within other models. During processing, these references would be replaced by actual values sampled from their corresponding distributions.
    """

    ref: ValueName
    """The unique identifier for the variable to be resolved."""


class NamedValueState(StrEnum):
    """Internal state of a NamedValue."""

    UNSET = "unset"
    """The value has been initialized but not yet sampled/populated."""

    SET = "set"
    """The value has been populated and is effectively frozen."""


def _numpy_safe_sentinel(v: Any) -> Any:
    """
    Prevents NumPy from blowing up during Pydantic Union validation.
    Checks identity/type before allowing a comparison to occur.
    """
    if isinstance(v, str) and v == UNSET_SENTINEL:
        return UNSET_SENTINEL
    return v


class NamedValue[T](BaseModel, NumericMixin):
    """
    A container for a sampled variable that tracks its initialization state.

    NamedValues act as the bridge between abstract distributions and concrete simulation parameters. They utilize a state-machine logic to ensure that values are not accidentally overwritten once sampled, and provide a NumericMixin to allow the container to behave like the underlying data in mathematical operations.

    Notes:
        Once `state` becomes `SET`, the `value` property will raise an error
        on further assignment attempts unless `force_set_value` is used.

    """

    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid",
        arbitrary_types_allowed=True,
    )

    name: ValueName
    """The unique identifier for this parameter."""

    state: NamedValueState = Field(default=NamedValueState.UNSET)
    """The current lifecycle state (SET or UNSET)."""

    stored_value: Annotated[T | UnsetType, BeforeValidator(_numpy_safe_sentinel)] = (
        Field(default=UNSET_SENTINEL)
    )
    """The actual data held by this container."""

    @field_serializer("stored_value", mode="plain", when_used="always")
    def _serialize_value(self, v: Any, info: FieldSerializationInfo) -> Any:
        """
        Silences the Pydantic Union warnings by providing a single-path serializer.

        Use to_jsonable_python to ensure numpy arrays (and any other T) are correctly converted to JSON primitives.
        """
        if isinstance(v, str) and v == UNSET_SENTINEL:
            return UNSET_SENTINEL

        # 2. Handle NumPy / Array-like types
        if hasattr(v, "tolist"):
            # If model_dump_json() or mode='json', we MUST return a list
            if info.mode == "json":
                return v.tolist()  # pyright: ignore[reportAttributeAccessIssue]
            # If model_dump() or mode='python', return the raw array for tests
            return v

        # 3. Handle everything else (floats, ints, etc.)
        return to_jsonable_python(v) if info.mode == "json" else v

    @model_validator(mode="after")
    def validate_state(self) -> Self:
        """Synchronizes the state enum with the actual stored_value content."""
        is_sentinel = (
            isinstance(self.stored_value, str) and self.stored_value == UNSET_SENTINEL
        )

        match self.state:
            case NamedValueState.UNSET:
                if not is_sentinel:
                    self.state = NamedValueState.SET
            case NamedValueState.SET:
                if is_sentinel:
                    msg = f"{self.name} stored value cannot be set to `NamedValueState.UNSET`"
                    logger.error(msg)
                    raise ValueError(msg)
            case _:
                msg = f"The enumeration for {self.state} has not been implemented."
                logger.error(msg)
                raise NotImplementedError(msg)
        return self

    @property
    def value(self) -> T:
        """
        Returns the stored value if it has been set.

        Raises:
            ValueError: If the state is UNSET.
            RuntimeError: If the internal state is corrupted.

        """
        is_sentinel = (
            isinstance(self.stored_value, str) and self.stored_value == UNSET_SENTINEL
        )

        match self.state:
            case NamedValueState.UNSET:
                msg = f"Value for NamedValue {self.name} has not been set."
                logger.error(msg)
                raise ValueError(msg)
            case NamedValueState.SET:
                if is_sentinel:
                    # Defensive: impossible unless model was corrupted
                    msg = f"NamedValue '{self.name}' is set but stored_value is unset implying something was corrupted!"
                    logger.error(msg)
                    raise RuntimeError(msg)
                return cast(T, self.stored_value)
            case _:
                msg = f"The enumeration for {self.state} has not been implemented."
                logger.error(msg)
                raise NotImplementedError(msg)

    @value.setter
    def value(self, value: T) -> None:
        """
        Sets the stored value and transitions state to SET.

        Raises:
            ValueError: If the value is already SET (frozen).

        """
        match self.state:
            case NamedValueState.SET:
                msg = f"Value for NamedValue {self.name} has already been set and is frozen."
                logger.error(msg)
                raise ValueError(msg)
            case NamedValueState.UNSET:
                self.force_set_value(value=value, warn=False)
            case _:
                msg = f"The enumeration for {self.state} has not been implemented."
                logger.error(msg)
                raise NotImplementedError(msg)

    def force_set_value(self, value: T, warn: bool = True) -> None:
        """
        Manually overrides the stored value regardless of current state.

        Args:
            value: The new value to store.
            warn: If True, logs a warning about the override.

        """
        if warn:
            logger.warning(f"Forcing value of NamedValue {self.name} to {value}")
        self.stored_value = value
        self.state = NamedValueState.SET

    @property
    def is_set(self) -> bool:
        """True if the value has been populated."""
        return self.state is NamedValueState.SET


class NamedValueDict[T](BaseDict[NamedValue[T]]):
    """Dictionary specifically for sampled results."""

    def __contains__(self, key: object) -> bool:
        if isinstance(key, NamedValue):
            key = key.name
        return super().__contains__(key=key)

    def get_value(self, name: ValueName | str) -> Any:
        """Gets the NamedValue value of a key."""
        return self[name].value

    def get_raw_value(self, name: ValueName | str) -> Any | None:
        """Gets the NamedValue raw value. This includes None if the value has not yet been set."""
        return self[name].stored_value

    @property
    def named_value_list(self) -> NamedValueList:
        """Converts the NamedValueDict to a NamedValueList."""
        return NamedValueList(list(self.values()))


class NamedValueList[T](BaseList[NamedValue[T]]):
    """List specifically for sampled results."""

    @property
    def to_named_value_dict(self) -> NamedValueDict:
        """Converts the NamedValueList to a NamedValueDict."""
        d = NamedValueDict()
        d.update_many(self.root)
        return d
