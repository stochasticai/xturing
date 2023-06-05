from typing import Any, Dict


class BaseParent:
    registry: Dict[str, Any] = {}

    @classmethod
    def add_to_registry(cls, name: str, obj: Any):
        assert (
            name not in cls.registry
        ), f"Class {name} already exists in base class {cls.__name__} registry {cls.registry}"
        cls.registry[name] = obj

    @classmethod
    def create(cls, class_key, *args, **kwargs):
        return cls.registry[class_key](*args, **kwargs)

    @classmethod
    def __getitem__(cls, key):
        assert (
            key in cls.registry
        ), f"Class {key} not found in base class {cls.__name__} registry {cls.registry}"
        return cls.registry[key]
