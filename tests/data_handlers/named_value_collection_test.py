import pickle

import numpy as np
import pytest

from stochas import NamedValue, NamedValueDict, NamedValueList
from stochas.named_value import ValueName


def test_dict_getitem_missing_key():
    """Ensure __getitem__ raises KeyError for a missing key."""
    d = NamedValueDict[int]()

    with pytest.raises(KeyError, match="not found"):
        d["missing"]


def test_dict_iter_keys_and_items():
    """Ensure __iter__, keys, and items expose the dictionary's contents."""
    d = NamedValueDict[int]()
    nv1 = NamedValue[int](name=ValueName("a"), stored_value=1)
    nv2 = NamedValue[int](name=ValueName("b"), stored_value=2)
    d.update(nv1)
    d.update(nv2)

    assert list(d) == ["a", "b"]
    assert list(d.keys()) == ["a", "b"]
    assert list(d.items()) == [("a", nv1), ("b", nv2)]


def test_dict_force_update_many():
    """Ensure force_update_many overwrites existing keys without error."""
    d = NamedValueDict[int]()
    nv1 = NamedValue[int](name=ValueName("x"), stored_value=1)
    nv2 = NamedValue[int](name=ValueName("x"), stored_value=2)
    nv3 = NamedValue[int](name=ValueName("y"), stored_value=3)

    d.update(nv1)
    d.force_update_many([nv2, nv3], warn=False)

    assert d.get_value("x") == 2
    assert d.get_value("y") == 3


def test_dict_pickle_roundtrip():
    """Ensure __reduce__ preserves the generic type parameter when pickling and unpickling."""
    d = NamedValueDict[int]()
    d.update(NamedValue[int](name=ValueName("x"), stored_value=1))

    restored = pickle.loads(pickle.dumps(d))

    assert restored == d
    assert restored.__class__ is d.__class__


def test_list_array_conversion():
    """Ensure __array__ converts the list to a numpy array."""
    nv_list = NamedValueList[int]()
    nv_list.append(NamedValue[int](name=ValueName("a"), stored_value=1))
    nv_list.append(NamedValue[int](name=ValueName("b"), stored_value=2))

    arr = np.array(nv_list)
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (2,)


def test_list_setitem():
    """Ensure __setitem__ replaces the item at an index."""
    nv_list = NamedValueList[int]()
    a = NamedValue[int](name=ValueName("a"))
    b = NamedValue[int](name=ValueName("b"))
    nv_list.append(a)

    nv_list[0] = b
    assert nv_list[0] is b


def test_list_pickle_roundtrip():
    """Ensure __reduce__ preserves the generic type parameter when pickling and unpickling."""
    nv_list = NamedValueList[int]()
    nv_list.append(NamedValue[int](name=ValueName("a"), stored_value=1))

    restored = pickle.loads(pickle.dumps(nv_list))

    assert restored == nv_list
    assert restored.__class__ is nv_list.__class__


def test_dict_setitem_validation():
    """Ensure __setitem__ enforces the key matches the name."""
    d = NamedValueDict[int]()
    nv = NamedValue[int](name=ValueName("correct_name"))

    # Happy path
    d["correct_name"] = nv

    # Error path: mismatch
    with pytest.raises(ValueError, match="must match name"):
        d["wrong_key"] = nv


def test_dict_force_update():
    """Ensure force_update actually overwrites without error."""
    d = NamedValueDict[int]()
    nv1 = NamedValue[int](name=ValueName("x"), stored_value=1)
    nv2 = NamedValue[int](name=ValueName("x"), stored_value=2)

    d.update(nv1)
    d.force_update(nv2)  # Should not raise KeyError

    assert d.get_value("x") == 2
    assert len(d) == 1


def test_dict_delitem():
    """Ensure __delitem__ removes an existing key and raises for a missing one."""
    d = NamedValueDict[int]()
    nv = NamedValue[int](name=ValueName("x"), stored_value=1)
    d.update(nv)

    del d["x"]
    assert "x" not in d

    with pytest.raises(KeyError, match="not found"):
        del d["x"]


def test_dict_get():
    """Ensure get returns the value, the default, or None."""
    d = NamedValueDict[int]()
    nv = NamedValue[int](name=ValueName("x"), stored_value=1)
    d.update(nv)

    assert d.get("x") is nv
    assert d.get("missing") is None
    assert d.get("missing", nv) is nv


def test_dict_pop():
    """Ensure pop removes and returns the value, supports a default, and raises otherwise."""
    d = NamedValueDict[int]()
    nv = NamedValue[int](name=ValueName("x"), stored_value=1)
    d.update(nv)

    assert d.pop("x") is nv
    assert "x" not in d

    sentinel = NamedValue[int](name=ValueName("default"))
    assert d.pop("x", sentinel) is sentinel

    with pytest.raises(KeyError, match="not found"):
        d.pop("x")


def test_dict_popitem_and_clear():
    """Ensure popitem removes a key-value pair and clear empties the dictionary."""
    d = NamedValueDict[int]()
    nv = NamedValue[int](name=ValueName("x"), stored_value=1)
    d.update(nv)

    key, value = d.popitem()
    assert key == "x"
    assert value is nv
    assert len(d) == 0

    d.update(nv)
    d.clear()
    assert len(d) == 0


def test_dict_to_list_conversion():
    """Test the @property conversion to NamedValueList."""
    d = NamedValueDict[int]()
    d.update_many(
        [
            NamedValue[int](name=ValueName("a"), stored_value=1),
            NamedValue[int](name=ValueName("b"), stored_value=2),
        ]
    )

    nv_list = d.named_value_list
    assert isinstance(nv_list, NamedValueList)
    assert len(nv_list) == 2
    assert any(x.name == "a" for x in nv_list)


def test_list_indexing_and_slicing():
    """Verify list-like access works for both ints and slices."""
    nv_list = NamedValueList[int]()
    items = [
        NamedValue[int](name=ValueName("a")),
        NamedValue[int](name=ValueName("b")),
        NamedValue[int](name=ValueName("c")),
    ]
    nv_list.extend(items)

    # Integer index
    assert nv_list[0].name == "a"

    # Slice
    subset = nv_list[1:]
    assert isinstance(subset, list)
    assert len(subset) == 2
    assert subset[0].name == "b"


def test_list_delitem_and_pop():
    """Ensure list removal methods work correctly."""
    nv_list = NamedValueList[int]()
    nv = NamedValue[int](name=ValueName("a"))
    nv_list.append(nv)

    assert len(nv_list) == 1

    # Test del
    del nv_list[0]
    assert len(nv_list) == 0

    # Test pop
    nv_list.append(nv)
    popped = nv_list.pop()
    assert popped is nv
    assert len(nv_list) == 0


def test_dict_reversed():
    """Ensure __reversed__ iterates keys in reverse insertion order."""
    d = NamedValueDict[int]()
    d.update(NamedValue[int](name=ValueName("a"), stored_value=1))
    d.update(NamedValue[int](name=ValueName("b"), stored_value=2))

    assert list(reversed(d)) == ["b", "a"]


def test_list_reversed_and_contains():
    """Ensure __reversed__ and __contains__ behave like their list counterparts."""
    nv_list = NamedValueList[int]()
    a = NamedValue[int](name=ValueName("a"))
    b = NamedValue[int](name=ValueName("b"))
    nv_list.extend([a, b])

    assert list(reversed(nv_list)) == [b, a]
    assert a in nv_list
    assert NamedValue[int](name=ValueName("c")) not in nv_list


def test_list_add_and_iadd():
    """Ensure __add__ and __iadd__ concatenate lists."""
    a = NamedValue[int](name=ValueName("a"))
    b = NamedValue[int](name=ValueName("b"))
    c = NamedValue[int](name=ValueName("c"))

    nv_list1 = NamedValueList[int]([a])
    nv_list2 = NamedValueList[int]([b])

    combined = nv_list1 + nv_list2
    assert isinstance(combined, NamedValueList)
    assert list(combined) == [a, b]
    assert list(nv_list1) == [a]  # Original is unchanged

    nv_list1 += [c]
    assert list(nv_list1) == [a, c]


def test_dict_setdefault():
    """Ensure setdefault returns the existing value or inserts and returns the default."""
    d = NamedValueDict[int]()
    nv = NamedValue[int](name=ValueName("x"), stored_value=1)
    d.update(nv)

    assert d.setdefault("x", nv) is nv

    default = NamedValue[int](name=ValueName("y"), stored_value=2)
    assert d.setdefault("y", default) is default
    assert d["y"] is default


def test_list_insert_remove_clear():
    """Ensure insert, remove, and clear behave like their list counterparts."""
    nv_list = NamedValueList[int]()
    a = NamedValue[int](name=ValueName("a"))
    b = NamedValue[int](name=ValueName("b"))
    nv_list.extend([a, b])

    nv_list.insert(0, NamedValue[int](name=ValueName("first")))
    assert nv_list[0].name == "first"

    nv_list.remove(a)
    assert a not in nv_list

    nv_list.clear()
    assert len(nv_list) == 0


def test_list_index_and_count():
    """Ensure index and count behave like their list counterparts."""
    nv_list = NamedValueList[int]()
    a = NamedValue[int](name=ValueName("a"))
    b = NamedValue[int](name=ValueName("b"))
    nv_list.extend([a, b, a])

    assert nv_list.index(b) == 1
    assert nv_list.count(a) == 2
    assert nv_list.count(b) == 1


def test_list_sort_and_reverse():
    """Ensure sort and reverse behave like their list counterparts."""
    nv_list = NamedValueList[int]()
    nv_list.extend(
        [
            NamedValue[int](name=ValueName("b")),
            NamedValue[int](name=ValueName("a")),
            NamedValue[int](name=ValueName("c")),
        ]
    )

    nv_list.sort(key=lambda nv: nv.name)
    assert [nv.name for nv in nv_list] == ["a", "b", "c"]

    nv_list.reverse()
    assert [nv.name for nv in nv_list] == ["c", "b", "a"]


def test_list_find_by_name():
    """Test the name lookup helper in the list."""
    nv_list = NamedValueList[str]()
    nv = NamedValue[str](name=ValueName("target"), stored_value="hit")
    nv_list.append(nv)

    assert nv_list.find_by_name("target").value == "hit"

    with pytest.raises(KeyError, match="not found in list"):
        nv_list.find_by_name("ghost")


def test_list_to_dict_conversion():
    """Verify ordered list can transform into a keyed dict."""
    nv_list = NamedValueList[int]()
    nv_list.append(NamedValue[int](name=ValueName("x"), stored_value=100))

    d = nv_list.to_named_value_dict
    assert isinstance(d, NamedValueDict)
    assert d.get_value("x") == 100


def test_list_to_dict_duplicate_fail():
    """List to Dict conversion should fail if list contains duplicate names."""
    nv_list = NamedValueList[int]()
    nv_list.append(NamedValue[int](name=ValueName("dup")))
    nv_list.append(NamedValue[int](name=ValueName("dup")))  # Valid in list

    with pytest.raises(KeyError, match="already been registered"):
        _ = nv_list.to_named_value_dict


if __name__ == "__main__":
    test_dict_setitem_validation()
