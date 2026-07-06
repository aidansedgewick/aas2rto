from __future__ import annotations  # must be first import

import inspect
from typing import Callable, Type

from aas2rto.query_managers.base import BaseQueryManager


class RegistrationError(Exception):
    pass


class AlreadyRegisteredError(RegistrationError):
    pass


class NotAQueryManagerError(RegistrationError):
    pass


class MalformedQueryManagerError(RegistrationError):
    pass


class QueryManagerRegistry:

    ### Choose not to have mutable _registry = {} class attr and classmethod.
    ### Cannot have isolated registries

    def __init__(self) -> None:
        self._registry: dict[str, Type[BaseQueryManager]] = {}

    # NOT a class method - used as the decorator
    def register(self):
        def decorator(qm_class: Type[BaseQueryManager]):
            return self._register_query_manager(qm_class)  # Still return class type.

        return decorator

    def _register_query_manager(self, qm_class: Type[BaseQueryManager]):
        # Normal method "inline" syntax useful for tests
        if not issubclass(qm_class, BaseQueryManager):
            name = getattr(qm_class, "__name__", repr(qm_class))
            msg = (
                "QueryManagerRegistry can only register types that inherit "
                f"from BaseQueryManager, not '{name}'"
            )
            raise NotAQueryManagerError(msg)

        if inspect.isabstract(qm_class):
            abcs = "\n".join(f"    - '{m}'" for m in qm_class.__abstractmethods__)
            cls_name = qm_class.__name__
            raise MalformedQueryManagerError(
                f"'{cls_name}' has unimplemented abstract methods/properties:\n{abcs}"
            )

        qm_name = qm_class.name
        existing = self._registry.get(qm_name)
        if existing is not None and existing is not qm_class:
            msg = (
                f"QM '{qm_name}' ({qm_class}) is already registered"
                f" as {existing.__class__}"
            )
            raise AlreadyRegisteredError(msg)

        self._registry[qm_name] = qm_class
        return qm_class

    def get(self, qm_name: str) -> Type[BaseQueryManager] | None:
        return self._registry.get(qm_name)

    def all(self) -> dict[str, Type[BaseQueryManager]]:
        return self._registry

    def __contains__(self, qm_name: str) -> bool:
        return qm_name in self._registry

    def __len__(self) -> int:
        return len(self._registry)


qm_registry = QueryManagerRegistry()  # inistantiate HERE

# def _autoregister():
#    import importlib, pkgutil
#    for
