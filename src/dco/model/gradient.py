from typing import Callable
from numpy import float64
from numpy.typing import NDArray


def grad(
    f: Callable[[NDArray[float64]], float], backend: str
) -> Callable[[NDArray[float64]], NDArray[float64]]:
    if backend == "jax":
        from jax import config, grad, jit

        config.update("jax_platforms", "cpu")

        return jit(grad(f))

    elif backend == "autograd":
        from autograd import grad

        return grad(f)

    else:
        raise ValueError(f"Unsupported backend: {backend}")
