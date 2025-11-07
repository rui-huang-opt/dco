from logging import getLogger

logger = getLogger("dco.optimizer")

from abc import ABCMeta, abstractmethod
from numpy import float64, sqrt
from numpy.typing import NDArray
from topolink import NodeHandle
from .loss_function import LossFunction


class Optimizer(metaclass=ABCMeta):
    _SUBCLASSES: dict[str, type["Optimizer"]] = {}

    def __init_subclass__(cls, key: str, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._SUBCLASSES[key] = cls

    def __init__(self, node_id: str, gamma: float, graph_name: str):
        self._node_handle = NodeHandle(node_id, graph_name)
        self._gamma = gamma
        self._aux_var: dict[str, NDArray[float64]] = {}

    @classmethod
    def create(
        cls,
        node_id: str,
        gamma: float,
        graph_name: str = "default",
        key: str = "RAugDGM",
    ) -> "Optimizer":
        Subclass = cls._SUBCLASSES.get(key)
        if Subclass is None:
            raise ValueError(f"Algorithm '{key}' is not registered.")
        return Subclass(node_id, gamma, graph_name)

    @abstractmethod
    def init(self, x_i: NDArray[float64], loss_fn: LossFunction) -> None: ...

    @abstractmethod
    def step(
        self, x_i: NDArray[float64], loss_fn: LossFunction
    ) -> NDArray[float64]: ...


class DGD(Optimizer, key="DGD"):
    """
    Distributed Gradient Descent (DGD) algorithm.
    """

    def __init__(self, node_id: str, gamma: float, graph_name: str):
        super().__init__(node_id, gamma, graph_name)

        self._k = 0

    def init(self, x_i: NDArray[float64], loss_fn: LossFunction) -> None:
        if loss_fn.g_type != "zero":
            err_msg = "DGD only supports loss functions without regularization."
            logger.error(err_msg)
            raise ValueError(err_msg)

    def step(self, x_i: NDArray[float64], loss_fn: LossFunction) -> NDArray[float64]:
        w_x_i = self._node_handle.weighted_mix(x_i)
        gamma_bar = self._gamma / sqrt(self._k + 1)
        grad = loss_fn.grad(x_i)

        new_x_i = w_x_i - gamma_bar * grad
        self._k += 1

        return new_x_i


class EXTRA(Optimizer, key="EXTRA"):
    def __init__(self, node_id: str, gamma: float, graph_name: str):
        super().__init__(node_id, gamma, graph_name)

    def init(self, x_i: NDArray[float64], loss_fn: LossFunction) -> None:
        w_x_i = self._node_handle.weighted_mix(x_i)
        self._aux_var["grad"] = loss_fn.grad(x_i)
        self._aux_var["new_z_i"] = w_x_i - self._gamma * self._aux_var["grad"]

    def step(self, x_i: NDArray[float64], loss_fn: LossFunction) -> NDArray[float64]:
        new_x_i = loss_fn.prox(self._gamma, self._aux_var["new_z_i"])
        p_i = self._aux_var["new_z_i"] + new_x_i - x_i

        w_p_i = self._node_handle.weighted_mix(p_i)
        new_grad = loss_fn.grad(new_x_i)

        grad_diff = new_grad - self._aux_var["grad"]

        new_new_z_i = 0.5 * (p_i + w_p_i) - self._gamma * grad_diff

        self._aux_var["grad"] = new_grad
        self._aux_var["new_z_i"] = new_new_z_i

        return new_x_i


class NIDS(Optimizer, key="NIDS"):
    def __init__(self, node_id: str, gamma: float, graph_name: str):
        super().__init__(node_id, gamma, graph_name)

    def init(self, x_i: NDArray[float64], loss_fn: LossFunction) -> None:
        self._aux_var["grad"] = loss_fn.grad(x_i)
        self._aux_var["new_z_i"] = x_i - self._gamma * self._aux_var["grad"]

    def step(self, x_i: NDArray[float64], loss_fn: LossFunction) -> NDArray[float64]:
        new_x_i = loss_fn.prox(self._gamma, self._aux_var["new_z_i"])
        new_grad = loss_fn.grad(new_x_i)

        x_i_diff = new_x_i - x_i
        grad_diff = new_grad - self._aux_var["grad"]

        p_i = self._aux_var["new_z_i"] + x_i_diff - self._gamma * grad_diff

        w_p_i = self._node_handle.weighted_mix(p_i)

        new_new_z_i = 0.5 * (p_i + w_p_i)

        self._aux_var["grad"] = new_grad
        self._aux_var["new_z_i"] = new_new_z_i

        return new_x_i


class DIGing(Optimizer, key="DIGing"):
    def __init__(self, node_id: str, gamma: float, graph_name: str):
        super().__init__(node_id, gamma, graph_name)

    def init(self, x_i: NDArray[float64], loss_fn: LossFunction) -> None:
        if loss_fn.g_type != "zero":
            err_msg = "DIGing only supports loss functions without regularization."
            logger.error(err_msg)
            raise ValueError(err_msg)

        self._aux_var["grad"] = loss_fn.grad(x_i)
        self._aux_var["y_i"] = self._aux_var["grad"]

    def step(self, x_i: NDArray[float64], loss_fn: LossFunction) -> NDArray[float64]:
        w_x_i = self._node_handle.weighted_mix(x_i)

        new_x_i = w_x_i - self._gamma * self._aux_var["y_i"]
        new_grad = loss_fn.grad(new_x_i)

        w_y_i = self._node_handle.weighted_mix(self._aux_var["y_i"])

        new_y_i = w_y_i + new_grad - self._aux_var["grad"]

        self._aux_var["grad"] = new_grad
        self._aux_var["y_i"] = new_y_i

        return new_x_i


class AugDGM(Optimizer, key="AugDGM"):
    def __init__(self, node_id: str, gamma: float, graph_name: str):
        super().__init__(node_id, gamma, graph_name)

    def init(self, x_i: NDArray[float64], loss_fn: LossFunction) -> None:
        self._aux_var["grad"] = loss_fn.grad(x_i)
        self._aux_var["y_i"] = self._aux_var["grad"]

    def step(self, x_i: NDArray[float64], loss_fn: LossFunction) -> NDArray[float64]:
        s_i = x_i - self._gamma * self._aux_var["y_i"]

        new_z_i = self._node_handle.weighted_mix(s_i)

        new_x_i = loss_fn.prox(self._gamma, new_z_i)
        new_grad = loss_fn.grad(new_x_i)

        grad_diff = new_grad - self._aux_var["grad"]
        new_prox_diff = (new_z_i - new_x_i) / self._gamma

        p_i = self._aux_var["y_i"] + grad_diff + new_prox_diff

        w_p_i = self._node_handle.weighted_mix(p_i)

        new_y_i = w_p_i - new_prox_diff

        self._aux_var["grad"] = new_grad
        self._aux_var["y_i"] = new_y_i

        return new_x_i


from numpy import zeros_like


class RGT(Optimizer, key="RGT"):
    def __init__(self, node_id: str, gamma: float, graph_name: str):
        super().__init__(node_id, gamma, graph_name)

    def init(self, x_i: NDArray[float64], loss_fn: LossFunction) -> None:
        self._aux_var["y_i"] = zeros_like(x_i)

    def step(self, x_i: NDArray[float64], loss_fn: LossFunction) -> NDArray[float64]:
        p_i = x_i + self._aux_var["y_i"]

        w_p_i = self._node_handle.weighted_mix(p_i)

        new_z_i = w_p_i - self._gamma * loss_fn.grad(x_i) - self._aux_var["y_i"]
        new_x_i = loss_fn.prox(self._gamma, new_z_i)

        q_i = new_z_i - new_x_i + x_i

        w_q_i = self._node_handle.weighted_mix(q_i)

        new_y_i = self._aux_var["y_i"] - w_q_i + new_z_i

        self._aux_var["y_i"] = new_y_i

        return new_x_i


class WE(Optimizer, key="WE"):
    def __init__(self, node_id: str, gamma: float, graph_name: str):
        super().__init__(node_id, gamma, graph_name)

    def init(self, x_i: NDArray[float64], loss_fn: LossFunction) -> None:
        self._aux_var["y_i"] = zeros_like(x_i)

    def step(self, x_i: NDArray[float64], loss_fn: LossFunction) -> NDArray[float64]:
        p_i = x_i + self._aux_var["y_i"]

        w_p_i = self._node_handle.weighted_mix(p_i)

        new_z_i = w_p_i - self._gamma * loss_fn.grad(x_i) - self._aux_var["y_i"]
        new_x_i = loss_fn.prox(self._gamma, new_z_i)

        q_i = new_z_i - new_x_i + x_i

        w_q_i = self._node_handle.weighted_mix(q_i)

        new_y_i = self._aux_var["y_i"] - w_q_i + q_i

        self._aux_var["y_i"] = new_y_i

        return new_x_i


class RAugDGM(Optimizer, key="RAugDGM"):
    def __init__(self, node_id: str, gamma: float, graph_name: str):
        super().__init__(node_id, gamma, graph_name)

    def init(self, x_i: NDArray[float64], loss_fn: LossFunction) -> None:
        self._aux_var["y_i"] = zeros_like(x_i)
        self._aux_var["s_i"] = x_i - self._gamma * loss_fn.grad(x_i)

    def step(self, x_i: NDArray[float64], loss_fn: LossFunction) -> NDArray[float64]:
        p_i = self._aux_var["s_i"] + self._aux_var["y_i"]

        w_p_i = self._node_handle.weighted_mix(p_i)

        new_z_i = w_p_i - self._aux_var["y_i"]
        new_x_i = loss_fn.prox(self._gamma, new_z_i)
        new_s_i = new_x_i - self._gamma * loss_fn.grad(new_x_i)

        q_i = new_z_i - new_s_i + self._aux_var["s_i"]

        w_q_i = self._node_handle.weighted_mix(q_i)

        new_y_i = self._aux_var["y_i"] - w_q_i + new_z_i

        self._aux_var["s_i"] = new_s_i
        self._aux_var["y_i"] = new_y_i

        return new_x_i


class AtcWE(Optimizer, key="AtcWE"):
    def __init__(self, node_id: str, gamma: float, graph_name: str):
        super().__init__(node_id, gamma, graph_name)

    def init(self, x_i: NDArray[float64], loss_fn: LossFunction) -> None:
        self._aux_var["y_i"] = zeros_like(x_i)
        self._aux_var["s_i"] = x_i - self._gamma * loss_fn.grad(x_i)

    def step(self, x_i: NDArray[float64], loss_fn: LossFunction) -> NDArray[float64]:
        p_i = self._aux_var["s_i"] + self._aux_var["y_i"]

        w_p_i = self._node_handle.weighted_mix(p_i)

        new_z_i = w_p_i - self._aux_var["y_i"]
        new_x_i = loss_fn.prox(self._gamma, new_z_i)
        new_s_i = new_x_i - self._gamma * loss_fn.grad(new_x_i)

        q_i = new_z_i - new_s_i + self._aux_var["s_i"]

        w_q_i = self._node_handle.weighted_mix(q_i)

        new_y_i = self._aux_var["y_i"] - w_q_i + q_i

        self._aux_var["s_i"] = new_s_i
        self._aux_var["y_i"] = new_y_i

        return new_x_i
