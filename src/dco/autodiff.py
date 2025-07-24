from typing import Callable, Any
from numpy import float64
from numpy.typing import NDArray


def grad(
    f: Callable[[NDArray[float64]], Any], backend: str
) -> Callable[[NDArray[float64]], NDArray[float64]]:
    if backend == "jax":
        from jax import grad, jit, config, device_get

        config.update("jax_platforms", "cpu")

        raw_grad = jit(grad(f))

        def wrapped_grad(x: NDArray[float64]) -> NDArray[float64]:
            return device_get(raw_grad(x))

        return wrapped_grad

    elif backend == "autograd":
        from autograd import grad

        return grad(f)  # type: ignore

    else:
        raise ValueError(f"Unsupported backend: {backend}")
