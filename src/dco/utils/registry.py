from typing import Type, TypeVar, Callable, Dict, Any, Generic

T = TypeVar("T")


class Registry(Generic[T]):
    def __init__(self):
        self._registry: Dict[str, Type[T]] = {}

    def register(self, name: str) -> Callable[[Type[T]], Type[T]]:
        def decorator(cls: Type[T]) -> Type[T]:
            if name in self._registry:
                raise ValueError(f"Class with name '{name}' is already registered.")
            self._registry[name] = cls
            return cls

        return decorator

    def create(self, name: str, *args: Any, **kwargs: Any) -> T:
        if name not in self._registry:
            raise ValueError(f"Unknown {self.__class__.__name__}: {name}")
        return self._registry[name](*args, **kwargs)
