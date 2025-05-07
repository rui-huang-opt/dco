import numpy as np
from numpy.typing import NDArray
from typing import Callable, Any
from ..numeric import grad, Regularizer


class Model:
    def __init__(
        self,
        dim: int,
        f_i: Callable[[NDArray[np.float64]], Any],
        g_type: str = "zero",
        lam: float = 1.0,
        backend: str = "autograd",
    ):
        self._dim = dim
        self._f_i = f_i
        self._grad_f_i = grad(f_i, backend)
        self._g_type = g_type
        self._g = Regularizer.create(g_type, lam)

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def g_type(self) -> str:
        return self._g_type

    @g_type.setter
    def g_type(self, value: str):
        self._g = Regularizer.create(value, self._g.lam)
        self._g_type = value

    def set_g_lam(self, lam: float) -> None:
        self._g.lam = lam

    def grad_f_i(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return self._grad_f_i(x)

    def prox_g(self, tau: float, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return self._g.prox(tau, x)
