import time
import logging
from multiprocessing.synchronize import Event, Barrier
from numpy import float64
from numpy.typing import NDArray
from topolink import NodeHandle
from .optimizer import Optimizer
from .utils import Logger
from .model import Model


def solve_sync(
    name: str,
    model: Model,
    alpha: float,
    gamma: float,
    algorithm: str = "RAugDGM",
    max_iter: int = 1000,
    server_address: str | None = None,
    *args,
    **kwargs,
) -> NDArray[float64]:
    node_handle = NodeHandle(name, server_address=server_address)

    optimizer = Optimizer.create(
        algorithm,
        model,
        node_handle,
        alpha,
        gamma,
        *args,
        **kwargs,
    )

    logger = logging.getLogger(f"dco.sync")

    logger.info(
        f"Starting algorithm '{algorithm}' "
        f"with parameters: alpha={alpha}, gamma={gamma}."
    )

    logger.info(f"Initial state: {optimizer.x_i}")

    begin_time = time.perf_counter()

    for k in range(max_iter):
        optimizer.step()

    end_time = time.perf_counter()

    logger.info(f"Final state: {optimizer.x_i}")

    logger.info(
        f"Completed algorithm '{algorithm}' "
        f"after {max_iter} iterations, "
        f"in {end_time - begin_time:.6f} seconds."
    )

    return optimizer.x_i


def solve_async(
    model: Model,
    alpha: float,
    gamma: float,
    stop_event: Event,
    algorithm: str = "RAugDGM",
    wait_time: float = 0.0,
    sync_barrier: Barrier | None = None,
    global_stop_event: Event | None = None,
    logger: Logger | None = None,
    *args,
    **kwargs,
) -> NDArray[float64]:
    raise NotImplementedError(
        "Asynchronous solving is not implemented yet. "
        "Please use solve_sync for synchronous solving."
    )
