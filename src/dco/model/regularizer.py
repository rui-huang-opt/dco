import numpy as np
from numpy.linalg import norm
from numpy.typing import NDArray
from abc import ABCMeta, abstractmethod
from typing import Union
from ..utils import Registry


class Regularizer(metaclass=ABCMeta):
    def __init__(self, lam: Union[int, float]):
        if lam <= 0:
            raise ValueError("Lambda must be positive.")
        self._lam = lam

    @abstractmethod
    def __call__(self, x: NDArray[np.float64]) -> np.float64: ...

    @abstractmethod
    def prox(
        self, tau: Union[int, float], x: NDArray[np.float64]
    ) -> NDArray[np.float64]: ...

    @property
    def lam(self) -> Union[int, float]:
        return self._lam

    @lam.setter
    def lam(self, value: Union[int, float]):
        if value <= 0:
            raise ValueError("Lambda must be positive.")
        self._lam = value


registry = Registry[Regularizer]()


@registry.register("zero")
class Zero(Regularizer):
    def __init__(self, lam: Union[int, float]):
        super().__init__(lam)

    def __call__(self, x: np.ndarray) -> Union[int, float]:
        return 0

    def prox(self, tau: Union[int, float], x: np.ndarray) -> np.ndarray:
        return x


@registry.register("l1")
class L1(Regularizer):
    def __init__(self, lam: Union[int, float]):
        super().__init__(lam)

    def __call__(self, x: np.ndarray) -> Union[int, float]:
        return self._lam * norm(x, ord=1)

    def prox(self, tau: Union[int, float], x: np.ndarray) -> np.ndarray:
        return np.maximum(np.abs(x) - tau * self._lam, 0) * np.sign(x)
