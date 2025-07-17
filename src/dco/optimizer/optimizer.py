from numpy import float64, zeros, asarray
from numpy.typing import NDArray
from abc import ABCMeta, abstractmethod
from topolink import NodeHandle
from ..utils import Registry
from ..model import Model


class Optimizer(metaclass=ABCMeta):
    registry = Registry["Optimizer"]()

    def __init_subclass__(cls, key: str | None = None, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.registry.register(cls, key)

    def __init__(
        self,
        model: Model,
        node_handle: NodeHandle,
        alpha: float,
        gamma: float,
        z_i_init: NDArray[float64] | None,
    ):
        self._model = model
        self._node_handle = node_handle

        self._alpha = alpha
        self._gamma = gamma

        self._z_i = self.initialize_array(z_i_init, model.dim)
        self._x_i = model.prox_g(gamma, self._z_i)

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

    @classmethod
    def create(
        cls,
        key: str,
        model: Model,
        node_handle: NodeHandle,
        alpha: float,
        gamma: float,
        z_i_init: NDArray[float64] | None = None,
        *args,
        **kwargs,
    ):
        return cls.registry.create(
            key, model, node_handle, alpha, gamma, z_i_init, *args, **kwargs
        )
