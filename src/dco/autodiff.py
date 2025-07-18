from typing import Callable, Any
from numpy import float64
from numpy.typing import NDArray


def grad(
    f: Callable[[NDArray[float64]], Any], backend: str
) -> Callable[[NDArray[float64]], NDArray[float64]]:
    if backend == "jax":
        from jax import grad, jit, config

        config.update("jax_platforms", "cpu")

        return jit(grad(f))

    elif backend == "autograd":
        from autograd import grad
        import autograd.numpy as anp

        return grad(f)  # type: ignore

    else:
        raise ValueError(f"Unsupported backend: {backend}")
