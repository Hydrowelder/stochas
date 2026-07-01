"""
Defines numeric mixins so that classes with a value attribute can act like a number.
"""

from __future__ import annotations

from typing import Any, Protocol, Self, runtime_checkable

import numpy as np
from pydantic import BaseModel

from stochas.unit_system import UnitDescriptor

__all__ = ["MetadataMixin", "NumericMixin"]


@runtime_checkable
class HasValue(Protocol):
    """The contract: Any class using the mixin must have a .value property."""

    stored_value: Any

    @property
    def value(self) -> Any: ...

    def _extract(self, other) -> Any: ...

    def force_set_value(self, value, warn) -> None: ...


class NumericMixin:
    """
    A mixin that proxies operations to the .value property.

    This allows a NamedValue to behave like a float, int, str, or NDArray.
    """

    # --- NumPy Masquerade ---
    # These allow you to call NamedValue.shape just like a real array
    @property
    def shape(self: HasValue) -> tuple[int, ...]:
        return self.value.shape

    @property
    def ndim(self: HasValue) -> int:
        return self.value.ndim

    @property
    def dtype(self: HasValue) -> Any:
        return self.value.dtype

    @property
    def size(self: HasValue) -> int:
        return self.value.size

    def squeeze(self: HasValue, axis: int | tuple[int, ...] | None = None) -> Self:  # pyright: ignore[reportGeneralTypeIssues]
        """
        Removes axes of length one from the underlying array.
        """
        # 1. Perform the squeeze on the data
        squeezed_data = self.value.squeeze(axis=axis)

        # 2. Update the underlying container, bypassing any "frozen" state checks
        self.force_set_value(squeezed_data, warn=False)

        return self  # pyright: ignore[reportReturnType]

    # --- Type Casting ---
    def __int__(self: HasValue) -> int:
        return int(self.value)

    def __float__(self: HasValue) -> float:
        return float(self.value)

    def __bool__(self: HasValue) -> bool:
        return bool(self.value)

    # --- Array & Sequence Behavior ---
    def __len__(self: HasValue) -> int:
        return len(self.value)

    def __getitem__(self: HasValue, key):
        return self.value[key]

    def __contains__(self: HasValue, item: Any) -> bool:
        return item in self.value

    # This allows np.array(named_value) to work
    def __array__(self: HasValue, dtype=None, copy=None):
        return np.array(self.value, dtype=dtype, copy=copy)

    # --- Math Operations ---
    # We use a helper to extract the raw value from 'other'
    # if 'other' is also a NamedValue.
    def _extract(self, other):
        return getattr(other, "value", other)

    def __add__(self: HasValue, other):
        return self.value + self._extract(other)

    def __sub__(self: HasValue, other):
        return self.value - self._extract(other)

    def __mul__(self: HasValue, other):
        return self.value * self._extract(other)

    def __truediv__(self: HasValue, other):
        return self.value / self._extract(other)

    def __pow__(self: HasValue, other):
        return self.value ** self._extract(other)

    def __matmul__(self: HasValue, other):
        return self.value @ self._extract(other)

    # Reverse operations (e.g., 5.0 + NamedValue)
    def __radd__(self: HasValue, other):
        return self._extract(other) + self.value

    def __rsub__(self: HasValue, other):
        return self._extract(other) - self.value

    def __rmul__(self: HasValue, other):
        return self._extract(other) * self.value

    def __rtruediv__(self: HasValue, other):
        return self._extract(other) / self.value

    # --- Comparisons ---
    def __lt__(self: HasValue, other):
        return self.value < self._extract(other)

    def __le__(self: HasValue, other):
        return self.value <= self._extract(other)

    def __gt__(self: HasValue, other):
        return self.value > self._extract(other)

    def __ge__(self: HasValue, other):
        return self.value >= self._extract(other)


class MetadataMixin(BaseModel):
    """
    Adds metadata attributes to an object.

    Inherits from BaseModel (rather than being a plain mixin) so that pyright's pydantic `__init__` synthesis recognizes these fields as constructor kwargs on classes that mix this in; a plain mixin's annotations are visible as attributes but are not picked up as constructor parameters.
    """

    category: str = "uncategorized"
    """Category for value. In distributions this is used to set the table folder this distribution will be written to."""

    unit: UnitDescriptor | None = None
    """Unit descriptor for this value. Set via `model.u.<unit_name>` (e.g. `model.u.inch`) to enable automatic conversion on sampling. The conversion factor is excluded from serialization and must be re-populated by calling `StochasBase.with_unit_system()` after deserialization. `str(unit)` returns the unit name (e.g. `"inch"`)."""

    source: str | None = None
    """Provenance of the value, such as a paper, datasheet, or calibration run. Defaults to unset."""

    display_name: str | None = None
    """Human-readable label for use in plots, reports, and UIs, distinct from the variable's internal name. Defaults to unset."""

    comment: str = ""
    """Extra context to include alongside this parameter."""

    def metadata_dict(self) -> dict[str, Any]:
        """Returns this instance's metadata fields as a dict, for passing on to another MetadataMixin instance."""
        return {name: getattr(self, name) for name in MetadataMixin.__annotations__}
