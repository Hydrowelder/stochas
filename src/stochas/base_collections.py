from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Iterator
from typing import TYPE_CHECKING, Any, Self, cast, overload

if TYPE_CHECKING:
    from _typeshed import SupportsRichComparison

import numpy as np
from pydantic import Field, RootModel

from stochas.utils import _reduce_obj

logger = logging.getLogger(__name__)

__all__ = ["BaseDict", "BaseList"]


class BaseDict[T](RootModel[dict[str, T]]):
    root: dict[str, T] = Field(default_factory=dict)

    def __reduce__(self):
        return _reduce_obj(self)

    def __getitem__(self, key: str) -> T:
        """Get an item in the dictionary with the specified key."""
        if key not in self.root:
            msg = f"'{key}' not found."
            logger.error(msg)
            raise KeyError(msg)
        return self.root[key]

    def __setitem__(self, key: str, value: T) -> None:
        """Set the value of a single key-value pair."""
        name = getattr(value, "name", None)
        if key != name:
            msg = f"Key '{key}' must match name '{name}'"
            logger.error(msg)
            raise ValueError(msg)
        self.update(value)

    def __iter__(self) -> Iterator[str]:  # pyright: ignore[reportIncompatibleMethodOverride]
        """Create an iterable object."""
        return self.root.__iter__()

    def __len__(self) -> int:
        """Returns the number of elements in the dictionary."""
        return self.root.__len__()

    def __delitem__(self, key: str) -> None:
        """Delete the item with the specified key."""
        if key not in self.root:
            msg = f"'{key}' not found."
            logger.error(msg)
            raise KeyError(msg)
        del self.root[key]

    def __contains__(self, key: object) -> bool:
        """Specifies if a key already exists in the dictionary."""
        return self.root.__contains__(key)

    def __reversed__(self) -> Iterator[str]:
        """Iterates over the dictionary's keys in reverse insertion order."""
        return reversed(self.root)

    def get(self, key: str, default: T | None = None) -> T | None:
        """Returns the value for key if it exists, otherwise default."""
        return self.root.get(key, default)

    def pop(self, key: str, *default: T) -> T:
        """Removes the specified key and returns the corresponding value. Raises KeyError if the key is not found and no default is given."""
        if key not in self.root:
            if default:
                return default[0]
            msg = f"'{key}' not found."
            logger.error(msg)
            raise KeyError(msg)
        return self.root.pop(key)

    def popitem(self) -> tuple[str, T]:
        """Removes and returns the last inserted key-value pair as a tuple."""
        return self.root.popitem()

    def clear(self) -> None:
        """Removes all items from the dictionary."""
        self.root.clear()

    def setdefault(self, key: str, default: T) -> T:
        """Returns the value for key if it exists, otherwise inserts default and returns it."""
        if key in self.root:
            return self.root[key]
        self[key] = default
        return default

    def keys(self):
        """Returns the keys in the dictionary."""
        return self.root.keys()

    def values(self):
        """Returns the values in the dictionary."""
        return self.root.values()

    def items(self):
        """Returns the key-value pairs in the dictionary as tuples."""
        return self.root.items()

    def update(self, value: T) -> None:
        """Add a new dictionary key value pair. The key cannot already exist in the dictionary."""
        name = getattr(value, "name", None)
        if name in self.root:
            msg = f"{name} has already been registered."
            logger.error(msg)
            raise KeyError(msg)

        self.force_update(value=value, warn=False)

    def update_many(self, values: Iterable[T]) -> None:
        """Add many new dictionary key value pair. The keys cannot already exist in the dictionary."""
        for value in values:
            self.update(value=value)

    def force_update(self, value: T, warn: bool = True) -> None:
        """Forces a key-value pair into the dictionary. Overwrites existing key if it exists."""
        name = getattr(value, "name")
        if warn:
            logger.warning(f"Forcing {name} into {self.__class__.__name__}.")
        self.root[name] = value

    def force_update_many(self, values: Iterable[T], warn: bool = True) -> None:
        """Forces adding many new dictionary key value pair. Overwrites existing keys if they exist."""
        for value in values:
            self.force_update(value=value, warn=warn)


class BaseList[T](RootModel[list[T]]):
    root: list[T] = Field(default_factory=list)

    def __reduce__(self):
        return _reduce_obj(self)

    def __array__(self, dtype=None, copy=None) -> np.ndarray:
        return np.array(self.root, dtype=dtype, copy=copy)

    def __len__(self) -> int:
        """Returns the number of elements in the list."""
        return self.root.__len__()

    def __iter__(self) -> Iterator[T]:  # pyright: ignore[reportIncompatibleMethodOverride]
        """Create an iterable object."""
        return self.root.__iter__()

    def __reversed__(self) -> Iterator[T]:
        """Iterates over the list in reverse order."""
        return reversed(self.root)

    def __contains__(self, value: object) -> bool:
        """Specifies if a value already exists in the list."""
        return self.root.__contains__(value)

    @overload
    def __getitem__(self, index: int) -> T: ...
    @overload
    def __getitem__(self, index: slice) -> list[T]: ...

    def __getitem__(self, index: int | slice) -> T | list[T]:
        """Gets the items in the specified index/indices."""
        return self.root[index]

    def __setitem__(self, index: int, value: T) -> None:
        """Set the object at an index."""
        self.root[index] = value

    def __delitem__(self, index: int) -> None:
        """Delete an element from the list."""
        del self.root[index]

    def append(self, value: T) -> None:
        """Append an item to the list."""
        self.root.append(value)

    def extend(self, values: Iterable[T]) -> None:
        """Append many items to the list."""
        self.root.extend(values)

    def __add__(self, other: BaseList[T] | list[T]) -> Self:
        """Returns a new list with the contents of both lists concatenated."""
        other_values = other.root if isinstance(other, BaseList) else other
        return self.__class__(root=[*self.root, *other_values])

    def __iadd__(self, other: BaseList[T] | list[T]) -> Self:
        """Extends the list in place with the contents of another list."""
        self.extend(other.root if isinstance(other, BaseList) else other)
        return self

    def pop(self, index: int = -1) -> T:
        """Remove and return item at index (default last)."""
        return self.root.pop(index)

    def insert(self, index: int, value: T) -> None:
        """Insert an item before the given index."""
        self.root.insert(index, value)

    def remove(self, value: T) -> None:
        """Remove the first occurrence of a value."""
        self.root.remove(value)

    def clear(self) -> None:
        """Remove all items from the list."""
        self.root.clear()

    def index(self, value: T, start: int = 0, stop: int | None = None) -> int:
        """Return the index of the first occurrence of a value."""
        if stop is None:
            stop = len(self.root)
        return self.root.index(value, start, stop)

    def count(self, value: T) -> int:
        """Return the number of occurrences of a value."""
        return self.root.count(value)

    def reverse(self) -> None:
        """Reverse the list in place."""
        self.root.reverse()

    def sort(
        self,
        *,
        key: Callable[[T], SupportsRichComparison] | None = None,
        reverse: bool = False,
    ) -> None:
        """Sort the list in place."""
        cast("list[Any]", self.root).sort(key=key, reverse=reverse)

    def find_by_name(self, name: str) -> T:
        """Utility to find a specific named value within the list."""
        for item in self.root:
            item_name = getattr(item, "name", None)
            if item_name == name:
                return item
        msg = f"Item '{name}' not found in list."
        logger.error(msg)
        raise KeyError(msg)


if __name__ == "__main__":
    from typing import Any

    from stochas import NamedValue, NamedValueDict, NamedValueList, ValueName

    name = NamedValue[str](name=ValueName("name"), stored_value="john")
    age = NamedValue[int](name=ValueName("age"), stored_value=1)

    named_value_dict = NamedValueDict()
    named_value_dict.update_many([name, age])

    assert name == named_value_dict["name"]
    x: NamedValue[str] = named_value_dict["name"]

    # Not recommended. Converted from NamedValue[str] to NamedValue[Any]
    named_value_list = NamedValueList[Any]([name])

    # Recommended. Object retains NamedValue[int]
    named_value_list.append(age)

    breakpoint()
