from numpy import float64
from numpy.typing import NDArray
from typing import Callable, Any
from .autodiff import grad
from .regularizer import Regularizer


class LocalObjective:
    def __init__(
        self,
        dim: int,
        f_i: Callable[[NDArray[float64]], Any],
        g_type: str = "zero",
        lam: float = 1.0,
        backend: str = "autograd",
    ):
        self._dim = dim
        self._g_type = g_type

        self.f_i = f_i
        self.g = Regularizer.create(g_type, lam)
        self.grad_f_i = grad(f_i, backend)
        self.prox_g = self.g.prox

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def g_type(self) -> str:
        return self._g_type
