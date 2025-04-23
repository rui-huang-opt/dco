from typing import Type, TypeVar, Callable, Dict, Any, Generic

T = TypeVar("T")


class Registry(Generic[T]):
    def __init__(self):
        self._registry: Dict[str, Type[T]] = {}

    def register(self, cls: Type[T], key: str):
        if key in self._registry:
            raise ValueError(f"Class with name '{key}' is already registered.")
        self._registry[key] = cls

    def create(self, name: str, *args: Any, **kwargs: Any) -> T:
        if name not in self._registry:
            raise ValueError(f"Unknown {self.__class__.__name__}: {name}")
        return self._registry[name](*args, **kwargs)
