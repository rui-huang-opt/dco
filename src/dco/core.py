import os
import time
import numpy as np
import logging
from typing import List
from multiprocessing.synchronize import Event, Barrier
from gossip import Gossip
from .model import Model
from .algorithm import Algorithm

logging.basicConfig(level=logging.INFO)


class Solver:
    def __init__(
        self,
        model: Model,
        communicator: Gossip,
    ):
        self._model = model
        self._communicator = communicator
        self.time_list: List[float] = []

    def solve(
        self,
        algorithm_name: str,
        alpha: int | float,
        gamma: int | float,
        max_iter: int = 1000,
        *args,
        **kwargs,
    ):
        algorithm = Algorithm.create(
            algorithm_name,
            self._model,
            self._communicator,
            alpha,
            gamma,
            *args,
            **kwargs,
        )

        begin = time.perf_counter()

        for k in range(max_iter):
            algorithm.update_model()
            algorithm.perform_iteration(k)

        end = time.perf_counter()

        logging.info(
            f"algorithm: {algorithm_name}, "
            f"node: {self._communicator.name}, "
            f"elapsed time: {end - begin:.6f}s"
        )

    def solve_async(
        self,
        algorithm_name: str,
        alpha: int | float,
        gamma: int | float,
        stop_event: Event | None = None,
        sync_barrier: Barrier | None = None,
        sleep_time: float | None = None,
        *args,
        **kwargs,
    ):
        algorithm = Algorithm.create(
            algorithm_name,
            self._model,
            self._communicator,
            alpha,
            gamma,
            *args,
            **kwargs,
        )

        k = 0
        sync_barrier.wait()
        start_time = time.perf_counter()

        while not stop_event.is_set():
            self.time_list.append(time.perf_counter() - start_time)
            algorithm.update_model()
            algorithm.perform_iteration(k)
            k += 1
            if sleep_time is not None:
                time.sleep(sleep_time)

    def save_results(self, save_path: str):
        os.makedirs(save_path, exist_ok=True)
        np.save(
            os.path.join(save_path, f"node_{self._communicator.name}.npy"),
            self._model.x_i_history,
        )
