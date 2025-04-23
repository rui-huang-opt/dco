import numpy as np
from numpy.typing import NDArray


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
