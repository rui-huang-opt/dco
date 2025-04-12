import numpy as np
from autograd import grad
from numpy.typing import NDArray
from typing import Callable, Union
from .regularizer import registry


class Model:
    def __init__(
        self,
        dim: int,
        f_i: Callable[[NDArray[np.float64]], np.float64],
        g_type: str = "zero",
        record_history: bool = True,
        lam: Union[int, float] = 1,
    ):
        self.dim = dim
        self._f_i = f_i
        self._grad_f_i = grad(f_i)
        self._g_type = g_type
        self._g = registry.create(g_type, lam)

        self._x_i: NDArray[np.float64] | None = None

        self._record_history = record_history

        if record_history:
            self._x_i_history = []

    @property
    def g_type(self) -> str:
        return self._g_type

    @g_type.setter
    def g_type(self, value: str):
        self._g = registry.create(value)
        self._g_type = value

    @property
    def x_i(self) -> NDArray[np.float64]:
        if self._x_i is None:
            raise ValueError("Model is not fitted yet.")
        return self._x_i

    @x_i.setter
    def x_i(self, value: NDArray[np.float64]):
        if value.shape != (self.dim,):
            raise ValueError(f"Expected shape ({self.dim},), got {value.shape}.")
        self._x_i = value

        if self._record_history:
            self._x_i_history.append(value)

    @property
    def value(self) -> np.float64:
        if self._x_i is None:
            raise ValueError("Model is not fitted yet.")
        return self._f_i(self._x_i) + self._g(self._x_i)

    @property
    def x_i_history(self) -> NDArray[np.float64]:
        return np.array(self._x_i_history)

    def set_g_lam(self, lam: Union[int, float]) -> None:
        self._g.lam = lam

    def grad_f_i(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return self._grad_f_i(x)

    def prox_g(self, tau: float, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return self._g.prox(tau, x)
