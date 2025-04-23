import numpy as np
from numpy.linalg import norm
from numpy.typing import NDArray
from abc import ABCMeta, abstractmethod
from ..utils import Registry


class Regularizer(metaclass=ABCMeta):
    registry = Registry["Regularizer"]()

    def __init_subclass__(cls, key: str, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.registry.register(cls, key)

    def __init__(self, lam: int | float):
        if lam <= 0:
            raise ValueError("Lambda must be positive.")
        self._lam = lam

    @abstractmethod
    def __call__(self, x: NDArray[np.float64]) -> np.float64: ...

    @abstractmethod
    def prox(self, tau: int | float, x: NDArray[np.float64]) -> NDArray[np.float64]: ...

    @property
    def lam(self) -> int | float:
        return self._lam

    @lam.setter
    def lam(self, value: int | float):
        if value <= 0:
            raise ValueError("Lambda must be positive.")
        self._lam = value

    @classmethod
    def create(cls, key: str, lam: int | float, *args, **kwargs):
        return cls.registry.create(key, lam, *args, **kwargs)


class Zero(Regularizer, key="zero"):
    def __init__(self, lam: int | float):
        super().__init__(lam)

    def __call__(self, x: np.ndarray) -> int | float:
        return 0

    def prox(self, tau: int | float, x: np.ndarray) -> np.ndarray:
        return x


class L1(Regularizer, key="l1"):
    def __init__(self, lam: int | float):
        super().__init__(lam)

    def __call__(self, x: np.ndarray) -> int | float:
        return self._lam * norm(x, ord=1)

    def prox(self, tau: int | float, x: np.ndarray) -> np.ndarray:
        return np.maximum(np.abs(x) - tau * self._lam, 0) * np.sign(x)
