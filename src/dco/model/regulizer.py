import numpy as np
from numpy.linalg import norm
from numpy.typing import NDArray
from abc import ABCMeta, abstractmethod
from typing import Union
from ..utils import Registry


class Regulizer(metaclass=ABCMeta):
    @abstractmethod
    def __call__(self, x: NDArray[np.float64]) -> np.float64: ...

    @abstractmethod
    def prox(
        self, tau: Union[int, float], x: NDArray[np.float64]
    ) -> NDArray[np.float64]: ...


registry = Registry[Regulizer]()


@registry.register("zero")
class Zero(Regulizer):
    def __call__(self, x: np.ndarray) -> Union[int, float]:
        return 0

    def prox(self, tau: Union[int, float], x: np.ndarray) -> np.ndarray:
        return x


@registry.register("l1")
class L1(Regulizer):
    def __init__(self, lam: Union[int, float] = 1):
        self.lam = lam

    def __call__(self, x: np.ndarray) -> Union[int, float]:
        return self.lam * norm(x, ord=1)

    def __mul__(self, lam: Union[int, float]):
        return self.__class__(lam * self.lam)

    def __rmul__(self, lam: Union[int, float]):
        return self.__mul__(lam)

    def prox(self, tau: Union[int, float], x: np.ndarray) -> np.ndarray:
        return np.maximum(np.abs(x) - tau * self.lam, 0) * np.sign(x)
