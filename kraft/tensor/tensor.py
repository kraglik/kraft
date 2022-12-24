from typing import Type, Optional
from numpy import single
from abc import ABC

from kraft.device import Device, np, cp
from .utils import (
    zeros_like
)


class Tensor(ABC):
    def __init__(self, data, dtype=np.float32, owner=None, target='cpu', requires_grad=True) -> None:
        if target == 'cpu':
            self._data = self._to_numpy(data, dtype)
        else:
            self._data = self._to_cupy(data, dtype)

        self._dtype = dtype
        self._owner = owner
        self._requires_grad = requires_grad
        self._grad = None

        if requires_grad:
            self.zero_grad()

    def to(self, device: Device) -> 'Tensor':
        if device.is_cpu:
            return self.to_cpu()

        return self.to_gpu(index=device.index)

    def to_cpu(self) -> 'Tensor':
        pass

    def to_gpu(self, index: Optional[int] = None) -> 'Tensor':
        pass

    def numpy(self) -> np.ndarray:
        return np.ndarray(self._data)

    def zero_grad(self) -> None:
        self._grad = zeros_like(self)

    @property
    def dtype(self) -> Type[single]:
        return self._dtype

    @staticmethod
    def _to_numpy(data, dtype) -> np.ndarray:
        return np.ndarray(data, dtype=dtype)

    @staticmethod
    def _to_cupy(data, dtype) -> cp.ndarray:
        return cp.ndarray(data, dtype)
