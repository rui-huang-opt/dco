import numpy as np
from numpy.linalg import norm
from numpy.typing import NDArray
from abc import ABCMeta, abstractmethod
from .utils import Registry


class Regularizer(metaclass=ABCMeta):
    _registry = Registry["Regularizer"]()

    def __init_subclass__(cls, key: str, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._registry.register(cls, key)

    def __init__(self, lam: float):
        if lam <= 0:
            raise ValueError("Lambda must be positive.")
        self._lam = lam

    @abstractmethod
    def __call__(self, x: NDArray[np.float64]) -> np.float64: ...

    @abstractmethod
    def prox(self, tau: float, x: NDArray[np.float64]) -> NDArray[np.float64]: ...

    @property
    def lam(self) -> float:
        return self._lam

    @lam.setter
    def lam(self, value: float):
        if value <= 0:
            raise ValueError("Lambda must be positive.")
        self._lam = value

    @classmethod
    def create(cls, key: str, lam: float, *args, **kwargs):
        return cls._registry.create(key, lam, *args, **kwargs)


class Zero(Regularizer, key="zero"):
    def __init__(self, lam: float):
        super().__init__(lam)

    def __call__(self, x: np.ndarray) -> float:
        return 0

    def prox(self, tau: float, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return x


class L1(Regularizer, key="l1"):
    def __init__(self, lam: float):
        super().__init__(lam)

    def __call__(self, x: np.ndarray) -> np.float64:
        return (self._lam * norm(x, ord=1)).astype(np.float64)

    def prox(self, tau: float, x: NDArray[np.float64]) -> NDArray[np.float64]:
        threshold = tau * self._lam
        return np.multiply(np.sign(x), np.maximum(np.abs(x) - threshold, 0))
