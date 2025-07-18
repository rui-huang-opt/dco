from typing import Type, TypeVar, Callable, Dict, Any, Generic

T = TypeVar("T")


class Registry(Generic[T]):
    def __init__(self):
        self._registry: Dict[str, Type[T]] = {}

    def register(self, cls: Type[T], key: str | None):
        if key is None:
            return
        if key in self._registry:
            raise ValueError(f"Class with name '{key}' is already registered.")
        self._registry[key] = cls

    def create(self, key: str, *args: Any, **kwargs: Any) -> T:
        if key not in self._registry:
            raise ValueError(f"Unknown {self.__class__.__name__}: {key}")
        return self._registry[key](*args, **kwargs)
