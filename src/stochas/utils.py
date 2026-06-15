"""Helpers for pickling pydantic generic models while preserving their type parameters."""

import importlib
from typing import Any

_GENERIC_MARKER = "__pydantic_generic__"


def _describe_type(tp: Any) -> Any:
    """Encodes a type for pickling, preserving any pydantic generic parameterization."""
    metadata = getattr(tp, "__pydantic_generic_metadata__", None)
    if metadata and metadata["origin"] is not None:
        origin = metadata["origin"]
        args = tuple(_describe_type(arg) for arg in metadata["args"])
        return (_GENERIC_MARKER, origin.__module__, origin.__name__, args)
    return tp


def _resolve_type(description: Any) -> Any:
    """Decodes a type description produced by `_describe_type`."""
    if (
        isinstance(description, tuple)
        and len(description) == 4
        and description[0] == _GENERIC_MARKER
    ):
        _, module_name, class_name, args = description
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)
        resolved_args = [_resolve_type(arg) for arg in args]
        return (
            cls[tuple(resolved_args)]
            if len(resolved_args) > 1
            else cls[resolved_args[0]]
        )
    return description


def _reduce_obj(obj: Any) -> tuple[Any, ...]:
    """
    Builds a `__reduce__` tuple for a pydantic generic model, preserving its type
    parameters across pickling (e.g. when sending objects between multiprocessing workers).
    """
    cls = obj.__class__
    metadata = cls.__pydantic_generic_metadata__
    origin = metadata["origin"]
    if origin is not None:
        module_name, class_name = origin.__module__, origin.__name__
        type_args = tuple(_describe_type(arg) for arg in metadata["args"])
    else:
        module_name, class_name = cls.__module__, cls.__name__
        type_args = ()
    return (_reconstruct_obj, (module_name, class_name, type_args, obj.model_dump()))


def _reconstruct_obj(
    module_name: str, class_name: str, type_args: tuple[Any, ...], data: Any
):
    """
    Dynamically imports the correct module and class, re-applies any generic type
    parameters, and uses Pydantic's universal loader to re-hydrate the object.
    """
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)

    if type_args:
        resolved_args = [_resolve_type(arg) for arg in type_args]
        cls = (
            cls[tuple(resolved_args)]
            if len(resolved_args) > 1
            else cls[resolved_args[0]]
        )

    return cls.model_validate(data)
