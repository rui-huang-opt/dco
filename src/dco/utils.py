from typing import Type, TypeVar, Dict, Any, Generic

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

    def get_class(self, key: str) -> Type[T]:
        if key not in self._registry:
            raise KeyError(f"Class '{key}' not found")
        return self._registry[key]

    def create(self, key: str, *args: Any, **kwargs: Any) -> T:
        cls = self.get_class(key)
        return cls(*args, **kwargs)
