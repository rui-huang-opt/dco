import time
import logging
from multiprocessing.synchronize import Event, Barrier
from numpy import float64, nan
from numpy.typing import NDArray
from gossip import Gossip
from .algorithm import Algorithm
from .utils import Logger
from .model import Model


def solve_sync(
    model: Model,
    communicator: Gossip,
    alpha: float,
    gamma: float,
    algorithm_name: str = "RAugDGM",
    max_iter: int = 1000,
    logger: Logger | None = None,
    *args,
    **kwargs,
) -> NDArray[float64]:
    algorithm = Algorithm.create(
        algorithm_name,
        model,
        communicator,
        alpha,
        gamma,
        *args,
        **kwargs,
    )

    begin_time = time.perf_counter()

    for k in range(max_iter):
        if logger is not None:
            logger.record_local(iteration=k, x_i=algorithm.x_i)

        algorithm.perform_iteration()

    end_time = time.perf_counter()

    logging.info(
        f"Node {communicator.name} completed algorithm '{algorithm_name}' "
        f"in {end_time - begin_time:.6f} seconds."
    )

    if logger is not None:
        logger.merge_local_to_global(f"node_{communicator.name}")

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
    algorithm = Algorithm.create(
        algorithm_name,
        model,
        communicator,
        alpha,
        gamma,
        *args,
        **kwargs,
    )

    if sync_barrier is not None:
        sync_barrier.wait()
    if logger is not None:
        logger.record_local(start_time=time.perf_counter())
    if wait_time > 0:
        time.sleep(wait_time)

    while not stop_event.is_set():
        if logger is not None:
            logger.record_local(timestamp=time.perf_counter(), x_i=algorithm.x_i)

        algorithm.perform_iteration()

        if global_stop_event is not None and global_stop_event.is_set():
            break

    if logger is not None:
        logger.record_local(timestamp=nan, x_i=algorithm.x_i)
        logger.merge_local_to_global(f"node_{communicator.name}")

    return algorithm.x_i
