import logging
from abc import ABC, abstractmethod, abstractproperty

import numpy as np

gpu_enabled = False
log = logging.getLogger("kraft.tensor")

try:
    import cupy as cp
    gpu_enabled = True
except ImportError:
    log.info("Failed to import CuPy - GPU support disabled")


class Tensor(ABC):
    def __init__(self, data, dtype=np.float32, owner=None, target='cpu') -> None:
        if target == 'cpu':
            self._data = self._to_numpy(data, dtype)
        else:
            self._data = self._to_cupy(data, dtype)

        self._owner = owner

    def to_cpu(self) -> 'Tensor':
        pass

    def to_gpu(self) -> 'Tensor':
        pass

    @staticmethod
    def _to_numpy(data, dtype) -> np.ndarray:
        return np.ndarray(data, dtype=dtype)

    @staticmethod
    def _to_cupy(data, dtype) -> cp.ndarray:
        return cp.ndarray(data, dtype)
