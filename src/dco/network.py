from typing import Protocol
from numpy import float64
from numpy.typing import NDArray


class NetworkOps(Protocol):
    """
    Protocol for communication operations in distributed optimization.

    Methods
    -------
    laplacian(state: NDArray[float64]) -> NDArray[float64]
        Compute the Laplacian of the input state across neighboring nodes.

    weighted_mix(state: NDArray[float64]) -> NDArray[float64]
        Perform a weighted mixing of the input state across neighboring nodes.
    """

    def laplacian(self, state: NDArray[float64]) -> NDArray[float64]: ...

    def weighted_mix(self, state: NDArray[float64]) -> NDArray[float64]: ...
