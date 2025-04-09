import numpy as np
from numpy.typing import NDArray
from abc import ABCMeta, abstractmethod
from gossip import Gossip
from ..utils import Registry, initialize_array
from ..model import Model


class Algorithm(metaclass=ABCMeta):
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


class LaplacianBased(Algorithm, metaclass=ABCMeta):
    def __init__(
        self,
        model: Model,
        communicator: Gossip,
        alpha: int | float,
        gamma: int | float,
        z_i_init: NDArray[np.float64] | None,
        **kwargs,
    ):
        super().__init__(model, communicator, alpha, gamma, z_i_init, **kwargs)

    def calculate_consensus_error(
        self, local_state: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        self._communicator.broadcast(local_state)
        neighbor_states = self._communicator.gather()

        return self._communicator.degree * local_state - sum(neighbor_states)


class RobustLaplacianBased(LaplacianBased, metaclass=ABCMeta):
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


registry = Registry[Algorithm]()


@registry.register("DGD")
class DGD(LaplacianBased):
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
        error_x_i = self.calculate_consensus_error(self._x_i, self._communicator)
        gamma_bar = self._gamma / (k + 1)
        grad_val = self._model.grad_f_i(self._x_i)

        self._x_i = self._x_i - self._alpha * error_x_i - gamma_bar * grad_val


@registry.register("EXTRA")
class EXTRA(LaplacianBased):
    def __init__(
        self,
        model: Model,
        communicator: Gossip,
        alpha: int | float,
        gamma: int | float,
        z_i_init: NDArray[np.float64] | None = None,
    ):
        super().__init__(model, communicator, alpha, gamma, z_i_init)

        error_x_i = self.calculate_consensus_error(self._x_i)

        self._grad_val = self._model.grad_f_i(self._x_i)
        self._new_z_i = (
            self._x_i - self._alpha * error_x_i - self._gamma * self._grad_val
        )

    def perform_iteration(self, k):
        new_x_i = self._model.prox_g(self._gamma, self._new_z_i)
        p_i = self._new_z_i + new_x_i - self._x_i

        error_p_i = self.calculate_consensus_error(p_i)
        new_grad_val = self._model.grad_f_i(new_x_i)

        new_new_z_i = (p_i - 0.5 * self._alpha * error_p_i) - self._gamma * (
            new_grad_val - self._grad_val
        )

        self._grad_val = new_grad_val
        self._x_i = new_x_i
        self._new_z_i = new_new_z_i


@registry.register("NIDS")
class NIDS(LaplacianBased):
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

        error_p_i = self.calculate_consensus_error(p_i)

        new_new_z_i = p_i - 0.5 * self._alpha * error_p_i

        self._grad_val = new_grad_val
        self._x_i = new_x_i
        self._new_z_i = new_new_z_i


@registry.register("DIGing")
class DIGing(LaplacianBased):
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
        error_x_i = self.calculate_consensus_error(self._x_i)

        new_x_i = self._x_i - self._alpha * error_x_i - self._gamma * self._y_i
        new_grad_val = self._model.grad_f_i(new_x_i)

        error_y_i = self.calculate_consensus_error(self._y_i)

        new_y_i = self._y_i - self._alpha * error_y_i + new_grad_val - self._grad_val

        self._grad_val = new_grad_val
        self._x_i = new_x_i
        self._y_i = new_y_i


@registry.register("AugDGM")
class AugDGM(LaplacianBased):
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

        error_s_i = self.calculate_consensus_error(s_i)

        new_z_i = s_i - self._alpha * error_s_i
        new_x_i = self._model.prox_g(self._gamma, new_z_i)
        new_grad_val = self._model.grad_f_i(new_x_i)

        p_i = self._y_i + new_grad_val - self._grad_val
        q_i = p_i + (new_z_i - new_x_i) / self._gamma

        error_q_i = self.calculate_consensus_error(q_i)

        new_y_i = p_i - self._alpha * error_q_i

        self._grad_val = new_grad_val
        self._x_i = new_x_i
        self._y_i = new_y_i


@registry.register("RGT")
class RGT(RobustLaplacianBased):
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

        error_p_i = self.calculate_consensus_error(p_i)

        new_z_i = (
            self._x_i
            - self._gamma * self._model.grad_f_i(self._x_i)
            - self._alpha * error_p_i
        )
        new_x_i = self._model.prox_g(self._gamma, new_z_i)

        q_i = new_x_i - self._x_i - new_z_i

        error_q_i = self.calculate_consensus_error(q_i)

        new_y_i = self._y_i + new_x_i - self._x_i - self._alpha * error_q_i

        self._x_i = new_x_i
        self._y_i = new_y_i


@registry.register("WE")
class WE(RobustLaplacianBased):
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

    def perform_iteration(self, k: int):
        p_i = self._x_i + self._y_i

        error_p_i = self.calculate_consensus_error(p_i)

        new_z_i = (
            self._x_i
            - self._gamma * self._model.grad_f_i(self._x_i)
            - self._alpha * error_p_i
        )
        new_x_i = self._model.prox_g(self._gamma, new_z_i)

        q_i = new_z_i - new_x_i + self._x_i

        error_q_i = self.calculate_consensus_error(q_i)

        new_y_i = self._y_i + self._alpha * error_q_i

        self._x_i = new_x_i
        self._y_i = new_y_i


@registry.register("RAugDGM")
class RAugDGM(RobustLaplacianBased):
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

    def perform_iteration(self, k: int):
        p_i = self._s_i + self._y_i

        error_p_i = self.calculate_consensus_error(p_i)

        new_z_i = self._s_i - self._alpha * error_p_i
        new_x_i = self._model.prox_g(self._gamma, new_z_i)
        new_s_i = new_x_i - self._gamma * self._model.grad_f_i(new_x_i)

        q_i = new_s_i - self._s_i
        t_i = q_i - new_z_i

        error_t_i = self.calculate_consensus_error(t_i)

        new_y_i = self._y_i + q_i - self._alpha * error_t_i

        self._x_i = new_x_i
        self._s_i = new_s_i
        self._y_i = new_y_i


@registry.register("AtcWE")
class AtcWE(RobustLaplacianBased):
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

    def perform_iteration(self, k: int):
        p_i = self._s_i + self._y_i

        error_p_i = self.calculate_consensus_error(p_i)

        new_z_i = self._s_i - self._alpha * error_p_i
        new_x_i = self._model.prox_g(self._gamma, new_z_i)
        new_s_i = new_x_i - self._gamma * self._model.grad_f_i(new_x_i)

        q_i = new_z_i - new_x_i + self._s_i

        error_q_i = self.calculate_consensus_error(q_i)

        new_y_i = self._y_i + self._alpha * error_q_i

        self._x_i = new_x_i
        self._s_i = new_s_i
        self._y_i = new_y_i


@registry.register("ADMM")
class ADMM(Algorithm):
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

    def perform_iteration(self, k: int):
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
