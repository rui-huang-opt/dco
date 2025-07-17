from numpy import float64
from numpy.typing import NDArray
from topolink import NodeHandle
from .optimizer import Optimizer
from ..model import Model


class DGD(Optimizer, key="DGD"):
    """
    Distributed Gradient Descent (DGD) algorithm.
    """

    def __init__(
        self,
        model: Model,
        node_handle: NodeHandle,
        alpha: float,
        gamma: float,
        z_i_init: NDArray[float64] | None = None,
    ):
        if model.g_type != "zero":
            raise ValueError("DIGing cannot be used for composite problems.")
        super().__init__(model, node_handle, alpha, gamma, z_i_init)
        self._k = 0

    def step(self):
        delta_x_i = self._node_handle.laplacian(self._x_i)
        gamma_bar = self._gamma / (self._k + 1)
        grad_val = self._model.grad_f_i(self._x_i)

        self._x_i = self._x_i - self._alpha * delta_x_i - gamma_bar * grad_val
        self._k += 1


class RGT(Optimizer, key="RGT"):
    def __init__(
        self,
        model: Model,
        node_handle: NodeHandle,
        alpha: float,
        gamma: float,
        z_i_init: NDArray[float64] | None = None,
        y_i_init: NDArray[float64] | None = None,
    ):
        super().__init__(model, node_handle, alpha, gamma, z_i_init)
        self._y_i = self.initialize_array(y_i_init, model.dim)

    def step(self):
        p_i = self._x_i + self._y_i

        delta_p_i = self._node_handle.laplacian(p_i)

        new_z_i = (
            self._x_i
            - self._gamma * self._model.grad_f_i(self._x_i)
            - self._alpha * delta_p_i
        )
        new_x_i = self._model.prox_g(self._gamma, new_z_i)

        q_i = new_x_i - self._x_i - new_z_i

        delta_q_i = self._node_handle.laplacian(q_i)

        new_y_i = self._y_i + new_x_i - self._x_i - self._alpha * delta_q_i

        self._x_i = new_x_i
        self._y_i = new_y_i


class WE(Optimizer, key="WE"):
    def __init__(
        self,
        model: Model,
        node_handle: NodeHandle,
        alpha: float,
        gamma: float,
        z_i_init: NDArray[float64] | None = None,
        y_i_init: NDArray[float64] | None = None,
    ):
        super().__init__(model, node_handle, alpha, gamma, z_i_init)
        self._y_i = self.initialize_array(y_i_init, model.dim)

    def step(self):
        p_i = self._x_i + self._y_i

        delta_p_i = self._node_handle.laplacian(p_i)

        new_z_i = (
            self._x_i
            - self._gamma * self._model.grad_f_i(self._x_i)
            - self._alpha * delta_p_i
        )
        new_x_i = self._model.prox_g(self._gamma, new_z_i)

        q_i = new_z_i - new_x_i + self._x_i

        delta_q_i = self._node_handle.laplacian(q_i)

        new_y_i = self._y_i + self._alpha * delta_q_i

        self._x_i = new_x_i
        self._y_i = new_y_i


class RAugDGM(Optimizer, key="RAugDGM"):
    def __init__(
        self,
        model: Model,
        node_handle: NodeHandle,
        alpha: float,
        gamma: float,
        z_i_init: NDArray[float64] | None = None,
        y_i_init: NDArray[float64] | None = None,
    ):
        super().__init__(model, node_handle, alpha, gamma, z_i_init)

        self._y_i = self.initialize_array(y_i_init, model.dim)
        self._s_i = self._x_i - self._gamma * self._model.grad_f_i(self._x_i)

    def step(self):
        p_i = self._s_i + self._y_i

        delta_p_i = self._node_handle.laplacian(p_i)

        new_z_i = self._s_i - self._alpha * delta_p_i
        new_x_i = self._model.prox_g(self._gamma, new_z_i)
        new_s_i = new_x_i - self._gamma * self._model.grad_f_i(new_x_i)

        q_i = new_s_i - self._s_i
        t_i = q_i - new_z_i

        delta_t_i = self._node_handle.laplacian(t_i)

        new_y_i = self._y_i + q_i - self._alpha * delta_t_i

        self._x_i = new_x_i
        self._s_i = new_s_i
        self._y_i = new_y_i


class AtcWE(Optimizer, key="AtcWE"):
    def __init__(
        self,
        model: Model,
        node_handle: NodeHandle,
        alpha: float,
        gamma: float,
        z_i_init: NDArray[float64] | None = None,
        y_i_init: NDArray[float64] | None = None,
    ):
        super().__init__(model, node_handle, alpha, gamma, z_i_init)

        self._y_i = self.initialize_array(y_i_init, model.dim)
        self._s_i = self._x_i - self._gamma * self._model.grad_f_i(self._x_i)

    def step(self):
        p_i = self._s_i + self._y_i

        delta_p_i = self._node_handle.laplacian(p_i)

        new_z_i = self._s_i - self._alpha * delta_p_i
        new_x_i = self._model.prox_g(self._gamma, new_z_i)
        new_s_i = new_x_i - self._gamma * self._model.grad_f_i(new_x_i)

        q_i = new_z_i - new_s_i + self._s_i

        delta_q_i = self._node_handle.laplacian(q_i)

        new_y_i = self._y_i + self._alpha * delta_q_i

        self._x_i = new_x_i
        self._s_i = new_s_i
        self._y_i = new_y_i
