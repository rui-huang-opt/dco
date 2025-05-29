import os
import numpy as np
from multiprocessing import Manager
from multiprocessing.managers import DictProxy, ListProxy
from numpy.typing import NDArray


class Logger:
    def __init__(self):
        self._manager = Manager()
        self._log: DictProxy[
            str, DictProxy[str, ListProxy[float | NDArray[np.float64]]]
        ] = self._manager.dict()
        self._local_log: dict[str, list[float | NDArray[np.float64]]] = {}

    def export_log(self) -> dict[str, dict[str, NDArray[np.float64]]]:
        log = {}
        for name, data in self._log.items():
            log[name] = {key: np.array(val) for key, val in data.items()}
        self._log.clear()
        return log

    def record_local(self, **value: NDArray[np.float64] | float):
        for key, val in value.items():
            if key not in self._local_log:
                self._local_log[key] = []

            self._local_log[key].append(val)

    def merge_local_to_global(self, name: str):
        if name not in self._log:
            self._log[name] = self._manager.dict()

        for key, val in self._local_log.items():
            if key not in self._log[name]:
                self._log[name][key] = self._manager.list()

            self._log[name][key].extend(val)
            val.clear()

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        log = self.export_log()
        for name, data in log.items():
            file_path = os.path.join(path, f"{name}.npz")
            np.savez(file_path, **data)
