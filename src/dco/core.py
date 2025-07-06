import time
import logging
from multiprocessing.synchronize import Event, Barrier
from numpy import float64, nan
from numpy.typing import NDArray
from topolink import NodeHandle
from gossip import Gossip
from .algorithm import Algorithm
from .utils import Logger
from .model import Model


def solve_sync(
    name: str,
    model: Model,
    alpha: float,
    gamma: float,
    algorithm_name: str = "RAugDGM",
    max_iter: int = 1000,
    server_address: str | None = None,
    *args,
    **kwargs,
) -> NDArray[float64]:
    node_handle = NodeHandle(name, server_address=server_address)

    algorithm = Algorithm.create(
        algorithm_name,
        model,
        node_handle,
        alpha,
        gamma,
        *args,
        **kwargs,
    )

    logger = logging.getLogger("dco.solve_sync")

    logger.info(
        f"Node {name} starting algorithm '{algorithm_name}' "
        f"with parameters: alpha={alpha}, gamma={gamma}, max_iter={max_iter}."
    )

    logger.info(f"Node {name} initial state: x_i={algorithm.x_i}")

    begin_time = time.perf_counter()

    for k in range(max_iter):
        # logger.info(f"Node {name} at iteration {k}: state: {algorithm.x_i}")
        algorithm.perform_iteration()

    end_time = time.perf_counter()

    logger.info(
        f"Node {name} final state after {max_iter} iterations: x_i={algorithm.x_i}"
    )

    logger.info(
        f"Node {name} completed algorithm '{algorithm_name}' "
        f"in {end_time - begin_time:.6f} seconds."
    )

    return algorithm.x_i


def solve_async(
    model: Model,
    communicator: Gossip,
    alpha: float,
    gamma: float,
    stop_event: Event,
    algorithm_name: str = "RAugDGM",
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
