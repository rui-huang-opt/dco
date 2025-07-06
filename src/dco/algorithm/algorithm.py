import numpy as np
from numpy.typing import NDArray
from abc import ABCMeta, abstractmethod
from topolink import NodeHandle
from ..utils import Registry
from ..model import Model


class Algorithm(metaclass=ABCMeta):
    registry = Registry["Algorithm"]()

    def __init_subclass__(cls, key: str | None = None, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.registry.register(cls, key)

    def __init__(
        self,
        model: Model,
        node_handle: NodeHandle,
        alpha: float,
        gamma: float,
        z_i_init: NDArray[np.float64] | None,
    ):
        self._model = model
        self._node_handle = node_handle

        self._alpha = alpha
        self._gamma = gamma

        self._z_i = self.initialize_array(z_i_init, model.dim)
        self._x_i = model.prox_g(gamma, self._z_i)

    @staticmethod
    def initialize_array(
        array: NDArray[np.float64] | None, dimension: int
    ) -> NDArray[np.float64]:
        if array is None:
            initialized_array = np.zeros(dimension)
        elif array.shape == (dimension,):
            initialized_array = array
        else:
            raise ValueError(f"Input array must have dimension {dimension}.")

        return initialized_array

    @property
    def x_i(self) -> NDArray[np.float64]:
        # Ensure x_i is a numpy array, not a autograd/jax array
        return np.asarray(self._x_i)

    @abstractmethod
    def perform_iteration(self): ...

    @classmethod
    def create(
        cls,
        key: str,
        model: Model,
        node_handle: NodeHandle,
        alpha: float,
        gamma: float,
        z_i_init: NDArray[np.float64] | None = None,
        *args,
        **kwargs,
    ):
        return cls.registry.create(
            key, model, node_handle, alpha, gamma, z_i_init, *args, **kwargs
        )


class DGD(Algorithm, key="DGD"):
    """
    Distributed Gradient Descent (DGD) algorithm.
    """

    def __init__(
        self,
        model: Model,
        node_handle: NodeHandle,
        alpha: float,
        gamma: float,
        z_i_init: NDArray[np.float64] | None = None,
    ):
        if model.g_type != "zero":
            raise ValueError("DIGing cannot be used for composite problems.")
        super().__init__(model, node_handle, alpha, gamma, z_i_init)
        self._k = 0

    def perform_iteration(self):
        delta_x_i = self._node_handle.laplacian(self._x_i)
        gamma_bar = self._gamma / (self._k + 1)
        grad_val = self._model.grad_f_i(self._x_i)

        self._x_i = self._x_i - self._alpha * delta_x_i - gamma_bar * grad_val
        self._k += 1


class EXTRA(Algorithm, key="EXTRA"):
    def __init__(
        self,
        model: Model,
        node_handle: NodeHandle,
        alpha: float,
        gamma: float,
        z_i_init: NDArray[np.float64] | None = None,
    ):
        super().__init__(model, node_handle, alpha, gamma, z_i_init)

        delta_x_i = self._node_handle.laplacian(self._x_i)

        self._grad_val = self._model.grad_f_i(self._x_i)
        self._new_z_i = (
            self._x_i - self._alpha * delta_x_i - self._gamma * self._grad_val
        )

    def perform_iteration(self):
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


class NIDS(Algorithm, key="NIDS"):
    def __init__(
        self,
        model: Model,
        node_handle: NodeHandle,
        alpha: float,
        gamma: float,
        z_i_init: NDArray[np.float64] | None = None,
    ):
        super().__init__(model, node_handle, alpha, gamma, z_i_init)

        self._grad_val = self._model.grad_f_i(self._x_i)
        self._new_z_i = self._x_i - self._gamma * self._grad_val

    def perform_iteration(self):
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


class DIGing(Algorithm, key="DIGing"):
    def __init__(
        self,
        model: Model,
        node_handle: NodeHandle,
        alpha: float,
        gamma: float,
        z_i_init: NDArray[np.float64] | None = None,
    ):
        if model.g_type != "zero":
            raise ValueError("DIGing cannot be used for composite problems.")
        super().__init__(model, node_handle, alpha, gamma, z_i_init)

        self._grad_val = self._model.grad_f_i(self._x_i)
        self._y_i = self._grad_val

    def perform_iteration(self):
        delta_x_i = self._node_handle.laplacian(self._x_i)

        new_x_i = self._x_i - self._alpha * delta_x_i - self._gamma * self._y_i
        new_grad_val = self._model.grad_f_i(new_x_i)

        delta_y_i = self._node_handle.laplacian(self._y_i)

        new_y_i = self._y_i - self._alpha * delta_y_i + new_grad_val - self._grad_val

        self._grad_val = new_grad_val
        self._x_i = new_x_i
        self._y_i = new_y_i


class AugDGM(Algorithm, key="AugDGM"):
    def __init__(
        self,
        model: Model,
        node_handle: NodeHandle,
        alpha: float,
        gamma: float,
        z_i_init: NDArray[np.float64] | None = None,
    ):
        super().__init__(model, node_handle, alpha, gamma, z_i_init)

        self._grad_val = self._model.grad_f_i(self._x_i)
        self._y_i = self._grad_val

    def perform_iteration(self):
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


class RGT(Algorithm, key="RGT"):
    def __init__(
        self,
        model: Model,
        node_handle: NodeHandle,
        alpha: float,
        gamma: float,
        z_i_init: NDArray[np.float64] | None = None,
        y_i_init: NDArray[np.float64] | None = None,
    ):
        super().__init__(model, node_handle, alpha, gamma, z_i_init)
        self._y_i = self.initialize_array(y_i_init, model.dim)

    def perform_iteration(self):
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


class WE(Algorithm, key="WE"):
    def __init__(
        self,
        model: Model,
        node_handle: NodeHandle,
        alpha: float,
        gamma: float,
        z_i_init: NDArray[np.float64] | None = None,
        y_i_init: NDArray[np.float64] | None = None,
    ):
        super().__init__(model, node_handle, alpha, gamma, z_i_init)
        self._y_i = self.initialize_array(y_i_init, model.dim)

    def perform_iteration(self):
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


class RAugDGM(Algorithm, key="RAugDGM"):
    def __init__(
        self,
        model: Model,
        node_handle: NodeHandle,
        alpha: float,
        gamma: float,
        z_i_init: NDArray[np.float64] | None = None,
        y_i_init: NDArray[np.float64] | None = None,
    ):
        super().__init__(model, node_handle, alpha, gamma, z_i_init)

        self._y_i = self.initialize_array(y_i_init, model.dim)
        self._s_i = self._x_i - self._gamma * self._model.grad_f_i(self._x_i)

    def perform_iteration(self):
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


class AtcWE(Algorithm, key="AtcWE"):
    def __init__(
        self,
        model: Model,
        node_handle: NodeHandle,
        alpha: float,
        gamma: float,
        z_i_init: NDArray[np.float64] | None = None,
        y_i_init: NDArray[np.float64] | None = None,
    ):
        super().__init__(model, node_handle, alpha, gamma, z_i_init)

        self._y_i = self.initialize_array(y_i_init, model.dim)
        self._s_i = self._x_i - self._gamma * self._model.grad_f_i(self._x_i)

    def perform_iteration(self):
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
