from logging import getLogger
from time import perf_counter
from numpy import float64, zeros, asarray
from numpy.typing import NDArray
from abc import ABCMeta, abstractmethod
from topolink import NodeHandle
from .utils import Registry
from .local_objective import LocalObjective


class Optimizer(metaclass=ABCMeta):
    registry = Registry["Optimizer"]()

    def __init_subclass__(cls, key: str | None = None, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.registry.register(cls, key)

    def __init__(
        self,
        name: str,
        local_obj: LocalObjective,
        alpha: float,
        gamma: float,
        z_i_init: NDArray[float64] | None,
        server_address: str | None = None,
    ):
        self._node_handle = NodeHandle(name, server_address=server_address)
        self._local_obj = local_obj

        self._alpha = alpha
        self._gamma = gamma

        self._z_i = self.initialize_array(z_i_init, local_obj.dim)
        self._x_i = local_obj.prox_g(gamma, self._z_i)

    @staticmethod
    def initialize_array(
        array: NDArray[float64] | None, dimension: int
    ) -> NDArray[float64]:
        if array is None:
            initialized_array = zeros(dimension)
        elif array.shape == (dimension,):
            initialized_array = array
        else:
            raise ValueError(f"Input array must have dimension {dimension}.")

        return initialized_array

    @property
    def x_i(self) -> NDArray[float64]:
        # Ensure x_i is a numpy array, not a autograd/jax array
        return asarray(self._x_i)

    @abstractmethod
    def step(self): ...

    def solve_sync(self, max_iter: int = 1000):
        logger = getLogger(f"dco.sync")

        logger.info(
            f"Starting algorithm '{self.__name__}' "
            f"with parameters: alpha={self._alpha}, gamma={self._gamma}."
        )

        logger.info(f"Initial state: {self.x_i}")

        begin_time = perf_counter()

        for k in range(max_iter):
            self.step()

        end_time = perf_counter()

        logger.info(f"Final state: {self.x_i}")

        logger.info(
            f"Completed algorithm '{self.__name__}' "
            f"after {max_iter} iterations, "
            f"in {end_time - begin_time:.6f} seconds."
        )

    @classmethod
    def create(
        cls,
        key: str,
        name: str,
        local_obj: LocalObjective,
        alpha: float,
        gamma: float,
        z_i_init: NDArray[float64] | None = None,
        *args,
        **kwargs,
    ):
        return cls.registry.create(
            key, name, local_obj, alpha, gamma, z_i_init, *args, **kwargs
        )


class DGD(Optimizer, key="DGD"):
    """
    Distributed Gradient Descent (DGD) algorithm.
    """

    def __init__(
        self,
        name: str,
        local_obj: LocalObjective,
        alpha: float,
        gamma: float,
        z_i_init: NDArray[float64] | None = None,
    ):
        if local_obj.g_type != "zero":
            raise ValueError("DGD cannot be used for composite problems.")
        super().__init__(name, local_obj, alpha, gamma, z_i_init)
        self._k = 0

    def step(self):
        delta_x_i = self._node_handle.laplacian(self._x_i)
        gamma_bar = self._gamma / (self._k + 1)
        grad_val = self._local_obj.grad_f_i(self._x_i)

        self._x_i = self._x_i - self._alpha * delta_x_i - gamma_bar * grad_val
        self._k += 1


class EXTRA(Optimizer, key="EXTRA"):
    def __init__(
        self,
        name: str,
        local_obj: LocalObjective,
        alpha: float,
        gamma: float,
        z_i_init: NDArray[float64] | None = None,
    ):
        super().__init__(name, local_obj, alpha, gamma, z_i_init)

        delta_x_i = self._node_handle.laplacian(self._x_i)

        self._grad_val = self._local_obj.grad_f_i(self._x_i)
        self._new_z_i = (
            self._x_i - self._alpha * delta_x_i - self._gamma * self._grad_val
        )

    def step(self):
        new_x_i = self._local_obj.prox_g(self._gamma, self._new_z_i)
        p_i = self._new_z_i + new_x_i - self._x_i

        delta_p_i = self._node_handle.laplacian(p_i)
        new_grad_val = self._local_obj.grad_f_i(new_x_i)

        new_new_z_i = (p_i - 0.5 * self._alpha * delta_p_i) - self._gamma * (
            new_grad_val - self._grad_val
        )

        self._grad_val = new_grad_val
        self._x_i = new_x_i
        self._new_z_i = new_new_z_i


class NIDS(Optimizer, key="NIDS"):
    def __init__(
        self,
        name: str,
        local_obj: LocalObjective,
        alpha: float,
        gamma: float,
        z_i_init: NDArray[float64] | None = None,
    ):
        super().__init__(name, local_obj, alpha, gamma, z_i_init)

        self._grad_val = self._local_obj.grad_f_i(self._x_i)
        self._new_z_i = self._x_i - self._gamma * self._grad_val

    def step(self):
        new_x_i = self._local_obj.prox_g(self._gamma, self._new_z_i)
        new_grad_val = self._local_obj.grad_f_i(new_x_i)

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
        name: str,
        local_obj: LocalObjective,
        alpha: float,
        gamma: float,
        z_i_init: NDArray[float64] | None = None,
    ):
        if local_obj.g_type != "zero":
            raise ValueError("DIGing cannot be used for composite problems.")
        super().__init__(name, local_obj, alpha, gamma, z_i_init)

        self._grad_val = self._local_obj.grad_f_i(self._x_i)
        self._y_i = self._grad_val

    def step(self):
        delta_x_i = self._node_handle.laplacian(self._x_i)

        new_x_i = self._x_i - self._alpha * delta_x_i - self._gamma * self._y_i
        new_grad_val = self._local_obj.grad_f_i(new_x_i)

        delta_y_i = self._node_handle.laplacian(self._y_i)

        new_y_i = self._y_i - self._alpha * delta_y_i + new_grad_val - self._grad_val

        self._grad_val = new_grad_val
        self._x_i = new_x_i
        self._y_i = new_y_i


class AugDGM(Optimizer, key="AugDGM"):
    def __init__(
        self,
        name: str,
        local_obj: LocalObjective,
        alpha: float,
        gamma: float,
        z_i_init: NDArray[float64] | None = None,
    ):
        super().__init__(name, local_obj, alpha, gamma, z_i_init)

        self._grad_val = self._local_obj.grad_f_i(self._x_i)
        self._y_i = self._grad_val

    def step(self):
        s_i = self._x_i - self._gamma * self._y_i

        delta_s_i = self._node_handle.laplacian(s_i)

        new_z_i = s_i - self._alpha * delta_s_i
        new_x_i = self._local_obj.prox_g(self._gamma, new_z_i)
        new_grad_val = self._local_obj.grad_f_i(new_x_i)

        p_i = self._y_i + new_grad_val - self._grad_val
        q_i = p_i + (new_z_i - new_x_i) / self._gamma

        delta_q_i = self._node_handle.laplacian(q_i)

        new_y_i = p_i - self._alpha * delta_q_i

        self._grad_val = new_grad_val
        self._x_i = new_x_i
        self._y_i = new_y_i


class RGT(Optimizer, key="RGT"):
    def __init__(
        self,
        name: str,
        local_obj: LocalObjective,
        alpha: float,
        gamma: float,
        z_i_init: NDArray[float64] | None = None,
        y_i_init: NDArray[float64] | None = None,
    ):
        super().__init__(name, local_obj, alpha, gamma, z_i_init)
        self._y_i = self.initialize_array(y_i_init, local_obj.dim)

    def step(self):
        p_i = self._x_i + self._y_i

        delta_p_i = self._node_handle.laplacian(p_i)

        new_z_i = (
            self._x_i
            - self._gamma * self._local_obj.grad_f_i(self._x_i)
            - self._alpha * delta_p_i
        )
        new_x_i = self._local_obj.prox_g(self._gamma, new_z_i)

        q_i = new_x_i - self._x_i - new_z_i

        delta_q_i = self._node_handle.laplacian(q_i)

        new_y_i = self._y_i + new_x_i - self._x_i - self._alpha * delta_q_i

        self._x_i = new_x_i
        self._y_i = new_y_i


class WE(Optimizer, key="WE"):
    def __init__(
        self,
        name: str,
        local_obj: LocalObjective,
        alpha: float,
        gamma: float,
        z_i_init: NDArray[float64] | None = None,
        y_i_init: NDArray[float64] | None = None,
    ):
        super().__init__(name, local_obj, alpha, gamma, z_i_init)
        self._y_i = self.initialize_array(y_i_init, local_obj.dim)

    def step(self):
        p_i = self._x_i + self._y_i

        delta_p_i = self._node_handle.laplacian(p_i)

        new_z_i = (
            self._x_i
            - self._gamma * self._local_obj.grad_f_i(self._x_i)
            - self._alpha * delta_p_i
        )
        new_x_i = self._local_obj.prox_g(self._gamma, new_z_i)

        q_i = new_z_i - new_x_i + self._x_i

        delta_q_i = self._node_handle.laplacian(q_i)

        new_y_i = self._y_i + self._alpha * delta_q_i

        self._x_i = new_x_i
        self._y_i = new_y_i


class RAugDGM(Optimizer, key="RAugDGM"):
    def __init__(
        self,
        name: str,
        local_obj: LocalObjective,
        alpha: float,
        gamma: float,
        z_i_init: NDArray[float64] | None = None,
        y_i_init: NDArray[float64] | None = None,
    ):
        super().__init__(name, local_obj, alpha, gamma, z_i_init)

        self._y_i = self.initialize_array(y_i_init, local_obj.dim)
        self._s_i = self._x_i - self._gamma * self._local_obj.grad_f_i(self._x_i)

    def step(self):
        p_i = self._s_i + self._y_i

        delta_p_i = self._node_handle.laplacian(p_i)

        new_z_i = self._s_i - self._alpha * delta_p_i
        new_x_i = self._local_obj.prox_g(self._gamma, new_z_i)
        new_s_i = new_x_i - self._gamma * self._local_obj.grad_f_i(new_x_i)

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
        name: str,
        local_obj: LocalObjective,
        alpha: float,
        gamma: float,
        z_i_init: NDArray[float64] | None = None,
        y_i_init: NDArray[float64] | None = None,
    ):
        super().__init__(name, local_obj, alpha, gamma, z_i_init)

        self._y_i = self.initialize_array(y_i_init, local_obj.dim)
        self._s_i = self._x_i - self._gamma * local_obj.grad_f_i(self._x_i)

    def step(self):
        p_i = self._s_i + self._y_i

        delta_p_i = self._node_handle.laplacian(p_i)

        new_z_i = self._s_i - self._alpha * delta_p_i
        new_x_i = self._local_obj.prox_g(self._gamma, new_z_i)
        new_s_i = new_x_i - self._gamma * self._local_obj.grad_f_i(new_x_i)

        q_i = new_z_i - new_s_i + self._s_i

        delta_q_i = self._node_handle.laplacian(q_i)

        new_y_i = self._y_i + self._alpha * delta_q_i

        self._x_i = new_x_i
        self._s_i = new_s_i
        self._y_i = new_y_i
