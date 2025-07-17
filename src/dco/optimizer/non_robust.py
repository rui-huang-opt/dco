from numpy import float64
from numpy.typing import NDArray
from topolink import NodeHandle
from .optimizer import Optimizer
from ..model import Model


class EXTRA(Optimizer, key="EXTRA"):
    def __init__(
        self,
        model: Model,
        node_handle: NodeHandle,
        alpha: float,
        gamma: float,
        z_i_init: NDArray[float64] | None = None,
    ):
        super().__init__(model, node_handle, alpha, gamma, z_i_init)

        delta_x_i = self._node_handle.laplacian(self._x_i)

        self._grad_val = self._model.grad_f_i(self._x_i)
        self._new_z_i = (
            self._x_i - self._alpha * delta_x_i - self._gamma * self._grad_val
        )

    def step(self):
        new_x_i = self._model.prox_g(self._gamma, self._new_z_i)
        p_i = self._new_z_i + new_x_i - self._x_i

        delta_p_i = self._node_handle.laplacian(p_i)
        new_grad_val = self._model.grad_f_i(new_x_i)

        new_new_z_i = (p_i - 0.5 * self._alpha * delta_p_i) - self._gamma * (
            new_grad_val - self._grad_val
        )

        self._grad_val = new_grad_val
        self._x_i = new_x_i
        self._new_z_i = new_new_z_i


class NIDS(Optimizer, key="NIDS"):
    def __init__(
        self,
        model: Model,
        node_handle: NodeHandle,
        alpha: float,
        gamma: float,
        z_i_init: NDArray[float64] | None = None,
    ):
        super().__init__(model, node_handle, alpha, gamma, z_i_init)

        self._grad_val = self._model.grad_f_i(self._x_i)
        self._new_z_i = self._x_i - self._gamma * self._grad_val

    def step(self):
        new_x_i = self._model.prox_g(self._gamma, self._new_z_i)
        new_grad_val = self._model.grad_f_i(new_x_i)

        p_i = (
            self._new_z_i
            + new_x_i
            - self._x_i
            - self._gamma * (new_grad_val - self._grad_val)
        )

        delta_p_i = self._node_handle.laplacian(p_i)

        new_new_z_i = p_i - 0.5 * self._alpha * delta_p_i

        self._grad_val = new_grad_val
        self._x_i = new_x_i
        self._new_z_i = new_new_z_i


class DIGing(Optimizer, key="DIGing"):
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

        self._grad_val = self._model.grad_f_i(self._x_i)
        self._y_i = self._grad_val

    def step(self):
        delta_x_i = self._node_handle.laplacian(self._x_i)

        new_x_i = self._x_i - self._alpha * delta_x_i - self._gamma * self._y_i
        new_grad_val = self._model.grad_f_i(new_x_i)

        delta_y_i = self._node_handle.laplacian(self._y_i)

        new_y_i = self._y_i - self._alpha * delta_y_i + new_grad_val - self._grad_val

        self._grad_val = new_grad_val
        self._x_i = new_x_i
        self._y_i = new_y_i


class AugDGM(Optimizer, key="AugDGM"):
    def __init__(
        self,
        model: Model,
        node_handle: NodeHandle,
        alpha: float,
        gamma: float,
        z_i_init: NDArray[float64] | None = None,
    ):
        super().__init__(model, node_handle, alpha, gamma, z_i_init)

        self._grad_val = self._model.grad_f_i(self._x_i)
        self._y_i = self._grad_val

    def step(self):
        s_i = self._x_i - self._gamma * self._y_i

        delta_s_i = self._node_handle.laplacian(s_i)

        new_z_i = s_i - self._alpha * delta_s_i
        new_x_i = self._model.prox_g(self._gamma, new_z_i)
        new_grad_val = self._model.grad_f_i(new_x_i)

        p_i = self._y_i + new_grad_val - self._grad_val
        q_i = p_i + (new_z_i - new_x_i) / self._gamma

        delta_q_i = self._node_handle.laplacian(q_i)

        new_y_i = p_i - self._alpha * delta_q_i

        self._grad_val = new_grad_val
        self._x_i = new_x_i
        self._y_i = new_y_i
