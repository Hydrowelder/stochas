import importlib
from typing import Any


def _reconstruct_obj(module_name: str, class_name: str, data: Any):
    """
    Dynamically imports the correct module and class to re-hydrate objects in a worker process.
    """
    # find the module where the class actually lives
    module = importlib.import_module(module_name)

    # strip generic type parameters if present (e.g., 'NamedValueDict[NDArray]' -> 'NamedValueDict')
    # This handles generic classes like NamedValueDict[T], NamedValueList[T], etc.
    if "[" in class_name:
        base_class_name = class_name.split("[")[0]
    else:
        base_class_name = class_name

    # get the unspecialized base class
    cls = getattr(module, base_class_name)

    # use Pydantic's universal loader
    return cls.model_validate(data)
