import numpy as np
from numpy.typing import NDArray
from abc import ABCMeta, abstractmethod
from gossip import Gossip
from ..utils import Registry, initialize_array
from ..model import Model


class Algorithm(metaclass=ABCMeta):
    registry = Registry["Algorithm"]()

    def __init_subclass__(cls, key: str | None = None, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.registry.register(cls, key)

    def __init__(
        self,
        model: Model,
        communicator: Gossip,
        alpha: int | float,
        gamma: int | float,
        z_i_init: NDArray[np.float64] | None,
    ):
        self._model = model
        self._communicator = communicator

        self._alpha = alpha
        self._gamma = gamma

        self._z_i = initialize_array(z_i_init, model.dim)
        self._x_i = model.prox_g(gamma, self._z_i)

    def update_model(self):
        self._model.x_i = self._x_i

    @abstractmethod
    def perform_iteration(self, k: int): ...

    @classmethod
    def create(
        cls,
        key: str,
        model: Model,
        communicator: Gossip,
        alpha: int | float,
        gamma: int | float,
        z_i_init: NDArray[np.float64] | None = None,
        *args,
        **kwargs,
    ):
        return cls.registry.create(
            key, model, communicator, alpha, gamma, z_i_init, *args, **kwargs
        )


class RobustAlgorithm(Algorithm, metaclass=ABCMeta):
    def __init__(
        self,
        model: Model,
        communicator: Gossip,
        alpha: int | float,
        gamma: int | float,
        z_i_init: NDArray[np.float64] | None,
        y_i_init: NDArray[np.float64] | None,
    ):
        super().__init__(model, communicator, alpha, gamma, z_i_init)
        self._y_i = initialize_array(y_i_init, model.dim)


class DGD(Algorithm, key="DGD"):
    """
    Distributed Gradient Descent (DGD) algorithm.
    """

    def __init__(
        self,
        model: Model,
        communicator: Gossip,
        alpha: int | float,
        gamma: int | float,
        z_i_init: NDArray[np.float64] | None = None,
    ):
        if model.g_type != "zero":
            raise ValueError("DIGing cannot be used for composite problems.")
        super().__init__(model, communicator, alpha, gamma, z_i_init)

    def perform_iteration(self, k):
        delta_x_i = self._communicator.compute_laplacian(self._x_i)
        gamma_bar = self._gamma / (k + 1)
        grad_val = self._model.grad_f_i(self._x_i)

        self._x_i = self._x_i - self._alpha * delta_x_i - gamma_bar * grad_val


class EXTRA(Algorithm, key="EXTRA"):
    def __init__(
        self,
        model: Model,
        communicator: Gossip,
        alpha: int | float,
        gamma: int | float,
        z_i_init: NDArray[np.float64] | None = None,
    ):
        super().__init__(model, communicator, alpha, gamma, z_i_init)

        delta_x_i = self._communicator.compute_laplacian(self._x_i)

        self._grad_val = self._model.grad_f_i(self._x_i)
        self._new_z_i = (
            self._x_i - self._alpha * delta_x_i - self._gamma * self._grad_val
        )

    def perform_iteration(self, k):
        new_x_i = self._model.prox_g(self._gamma, self._new_z_i)
        p_i = self._new_z_i + new_x_i - self._x_i

        delta_p_i = self._communicator.compute_laplacian(p_i)
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
        communicator: Gossip,
        alpha: int | float,
        gamma: int | float,
        z_i_init: NDArray[np.float64] | None = None,
    ):
        super().__init__(model, communicator, alpha, gamma, z_i_init)

        self._grad_val = self._model.grad_f_i(self._x_i)
        self._new_z_i = self._x_i - self._gamma * self._grad_val

    def perform_iteration(self, k):
        new_x_i = self._model.prox_g(self._gamma, self._new_z_i)
        new_grad_val = self._model.grad_f_i(new_x_i)

        p_i = (
            self._new_z_i
            + new_x_i
            - self._x_i
            - self._gamma * (new_grad_val - self._grad_val)
        )

        delta_p_i = self._communicator.compute_laplacian(p_i)

        new_new_z_i = p_i - 0.5 * self._alpha * delta_p_i

        self._grad_val = new_grad_val
        self._x_i = new_x_i
        self._new_z_i = new_new_z_i


class DIGing(Algorithm, key="DIGing"):
    def __init__(
        self,
        model: Model,
        communicator: Gossip,
        alpha: int | float,
        gamma: int | float,
        z_i_init: NDArray[np.float64] | None = None,
    ):
        if model.g_type != "zero":
            raise ValueError("DIGing cannot be used for composite problems.")
        super().__init__(model, communicator, alpha, gamma, z_i_init)

        self._grad_val = self._model.grad_f_i(self._x_i)
        self._y_i = self._grad_val

    def perform_iteration(self, k):
        delta_x_i = self._communicator.compute_laplacian(self._x_i)

        new_x_i = self._x_i - self._alpha * delta_x_i - self._gamma * self._y_i
        new_grad_val = self._model.grad_f_i(new_x_i)

        delta_y_i = self._communicator.compute_laplacian(self._y_i)

        new_y_i = self._y_i - self._alpha * delta_y_i + new_grad_val - self._grad_val

        self._grad_val = new_grad_val
        self._x_i = new_x_i
        self._y_i = new_y_i


class AugDGM(Algorithm, key="AugDGM"):
    def __init__(
        self,
        model: Model,
        communicator: Gossip,
        alpha: int | float,
        gamma: int | float,
        z_i_init: NDArray[np.float64] | None = None,
    ):
        super().__init__(model, communicator, alpha, gamma, z_i_init)

        self._grad_val = self._model.grad_f_i(self._x_i)
        self._y_i = self._grad_val

    def perform_iteration(self, k):
        s_i = self._x_i - self._gamma * self._y_i

        delta_s_i = self._communicator.compute_laplacian(s_i)

        new_z_i = s_i - self._alpha * delta_s_i
        new_x_i = self._model.prox_g(self._gamma, new_z_i)
        new_grad_val = self._model.grad_f_i(new_x_i)

        p_i = self._y_i + new_grad_val - self._grad_val
        q_i = p_i + (new_z_i - new_x_i) / self._gamma

        delta_q_i = self._communicator.compute_laplacian(q_i)

        new_y_i = p_i - self._alpha * delta_q_i

        self._grad_val = new_grad_val
        self._x_i = new_x_i
        self._y_i = new_y_i


class RGT(RobustAlgorithm, key="RGT"):
    def __init__(
        self,
        model: Model,
        communicator: Gossip,
        alpha: int | float,
        gamma: int | float,
        z_i_init: NDArray[np.float64] | None = None,
        y_i_init: NDArray[np.float64] | None = None,
    ):
        super().__init__(model, communicator, alpha, gamma, z_i_init, y_i_init)

    def perform_iteration(self, k):
        p_i = self._x_i + self._y_i

        delta_p_i = self._communicator.compute_laplacian(p_i)

        new_z_i = (
            self._x_i
            - self._gamma * self._model.grad_f_i(self._x_i)
            - self._alpha * delta_p_i
        )
        new_x_i = self._model.prox_g(self._gamma, new_z_i)

        q_i = new_x_i - self._x_i - new_z_i

        delta_q_i = self._communicator.compute_laplacian(q_i)

        new_y_i = self._y_i + new_x_i - self._x_i - self._alpha * delta_q_i

        self._x_i = new_x_i
        self._y_i = new_y_i


class WE(RobustAlgorithm, key="WE"):
    def __init__(
        self,
        model: Model,
        communicator: Gossip,
        alpha: int | float,
        gamma: int | float,
        z_i_init: NDArray[np.float64] | None = None,
        y_i_init: NDArray[np.float64] | None = None,
    ):
        super().__init__(model, communicator, alpha, gamma, z_i_init, y_i_init)

    def perform_iteration(self, k):
        p_i = self._x_i + self._y_i

        delta_p_i = self._communicator.compute_laplacian(p_i)

        new_z_i = (
            self._x_i
            - self._gamma * self._model.grad_f_i(self._x_i)
            - self._alpha * delta_p_i
        )
        new_x_i = self._model.prox_g(self._gamma, new_z_i)

        q_i = new_z_i - new_x_i + self._x_i

        delta_q_i = self._communicator.compute_laplacian(q_i)

        new_y_i = self._y_i + self._alpha * delta_q_i

        self._x_i = new_x_i
        self._y_i = new_y_i


class RAugDGM(RobustAlgorithm, key="RAugDGM"):
    def __init__(
        self,
        model: Model,
        communicator: Gossip,
        alpha: int | float,
        gamma: int | float,
        z_i_init: NDArray[np.float64] | None = None,
        y_i_init: NDArray[np.float64] | None = None,
    ):
        super().__init__(model, communicator, alpha, gamma, z_i_init, y_i_init)

        self._s_i = self._x_i - self._gamma * self._model.grad_f_i(self._x_i)

    def perform_iteration(self, k):
        p_i = self._s_i + self._y_i

        delta_p_i = self._communicator.compute_laplacian(p_i)

        new_z_i = self._s_i - self._alpha * delta_p_i
        new_x_i = self._model.prox_g(self._gamma, new_z_i)
        new_s_i = new_x_i - self._gamma * self._model.grad_f_i(new_x_i)

        q_i = new_s_i - self._s_i
        t_i = q_i - new_z_i

        delta_t_i = self._communicator.compute_laplacian(t_i)

        new_y_i = self._y_i + q_i - self._alpha * delta_t_i

        self._x_i = new_x_i
        self._s_i = new_s_i
        self._y_i = new_y_i


class AtcWE(RobustAlgorithm, key="AtcWE"):
    def __init__(
        self,
        model: Model,
        communicator: Gossip,
        alpha: int | float,
        gamma: int | float,
        z_i_init: NDArray[np.float64] | None = None,
        y_i_init: NDArray[np.float64] | None = None,
    ):
        super().__init__(model, communicator, alpha, gamma, z_i_init, y_i_init)

        self._s_i = self._x_i - self._gamma * self._model.grad_f_i(self._x_i)

    def perform_iteration(self, k):
        p_i = self._s_i + self._y_i

        delta_p_i = self._communicator.compute_laplacian(p_i)

        new_z_i = self._s_i - self._alpha * delta_p_i
        new_x_i = self._model.prox_g(self._gamma, new_z_i)
        new_s_i = new_x_i - self._gamma * self._model.grad_f_i(new_x_i)

        q_i = new_z_i - new_s_i + self._s_i

        delta_q_i = self._communicator.compute_laplacian(q_i)

        new_y_i = self._y_i + self._alpha * delta_q_i

        self._x_i = new_x_i
        self._s_i = new_s_i
        self._y_i = new_y_i


class ADMM(Algorithm, key="ADMM"):
    def __init__(
        self,
        model: Model,
        communicator: Gossip,
        alpha: int | float,
        gamma: int | float,
        z_i_init: NDArray[np.float64] | None = None,
        rho: int | float = 0.5,
        delta: int | float = 0.5,
    ):
        if model.g_type != "zero":
            raise ValueError("DIGing cannot be used for composite problems.")
        super().__init__(model, communicator, alpha, gamma, z_i_init)

        self._rho = rho
        self._delta = delta
        self._beta = 1 + rho * self._communicator.degree
        self._zeta_i = {
            j: np.zeros(model.dim) for j in self._communicator.neighbor_names
        }

    def perform_iteration(self, k):
        y_s_stack_i = (
            np.hstack((self._x_i, self._model.grad_f_i(self._x_i))) + sum(self._zeta_i)
        ) / self._beta

        y_i = y_s_stack_i[: self._model.dim]
        s_i = y_s_stack_i[self._model.dim :]

        self._x_i = self._x_i + self._gamma * (y_i - self._x_i) - self._gamma * s_i

        for j in self._communicator.neighbor_names:
            m_ij = -self._zeta_i[j] + 2 * self._rho * y_s_stack_i

            self._communicator.send(j, m_ij)
            m_ji = self._communicator.recv(j)

            self._zeta_i[j] = (1 - self._delta) * self._zeta_i[j] + self._delta * m_ji
