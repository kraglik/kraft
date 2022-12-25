from typing import Any, Optional, Type

import numpy as np
from numpy import single

import kraft.device.utils
from kraft.device import Device


class Tensor(object):
    def __init__(
        self,
        data,
        requires_grad=True,
        device=None
    ):
        self.requires_grad = requires_grad
        self._device = device
        self.grad = None
        self.grad_fn = None

        if device.is_cpu:
            self.data = self._to_numpy_ndarray(data)
        else:
            self.data = self._to_cupy_ndarray(data, device)

    @property
    def device(self) -> Device:
        return self._device

    def detach(self) -> 'Tensor':
        return Tensor(self.data, requires_grad=True, device=self.device)

    def to(self, device: Device) -> 'Tensor':
        raise NotImplementedError

    def to_cpu(self) -> 'Tensor':
        raise NotImplementedError

    def to_gpu(self, index: Optional[int] = None) -> 'Tensor':
        raise NotImplementedError

    def numpy(self) -> np.ndarray:
        raise NotImplementedError

    def zero_grad(self) -> None:
        raise NotImplementedError

    @property
    def dtype(self) -> Type[single]:
        raise NotImplementedError

    @property
    def shape(self) -> tuple[int]:
        return self.data.shape

    def scalar(self) -> float:
        raise NotImplementedError

    def neg(self) -> 'Tensor':
        raise NotImplementedError

    def __matmul__(self, other: 'Tensor') -> 'Tensor':
        raise NotImplementedError

    def __abs__(self) -> 'Tensor':
        raise NotImplementedError

    def __add__(self, t):
        raise NotImplementedError

    def __radd__(self, t):
        raise NotImplementedError

    def __sub__(self, t):
        raise NotImplementedError

    def __rsub__(self, t):
        raise NotImplementedError

    def __mul__(self, t):
        raise NotImplementedError

    def __rmul__(self, t):
        raise NotImplementedError

    def __neg__(self):
        raise NotImplementedError

    def __truediv__(self, t):
        raise NotImplementedError

    def __rtruediv__(self, t):
        raise NotImplementedError

    def __pow__(self, t):
        raise NotImplementedError

    def __rpow__(self, t):
        raise NotImplementedError

    def __isub__(self, t):
        raise NotImplementedError

    def __eq__(self, other):
        raise NotImplementedError

    def __ne__(self, other):
        raise NotImplementedError

    def __lt__(self, other):
        raise NotImplementedError

    def __gt__(self, other):
        raise NotImplementedError

    def __le__(self, other):
        raise NotImplementedError

    def __ge__(self, other):
        raise NotImplementedError

    def __getitem__(self, item):
        raise NotImplementedError

    def __setitem__(self, key, value):
        raise NotImplementedError

    @staticmethod
    def _to_numpy_ndarray(data):
        return np.array(data)

    @staticmethod
    def _to_cupy_ndarray(data, device):
        if not kraft.device.utils.gpu_available:
            raise RuntimeError("Attempted to move tensor to GPU in a system without available GPU!")

        import cupy

        with cupy.cuda.Device(device.index):
            if isinstance(data, (np.ndarray, cupy.ndarray)):
                return cupy.asarray(data)

            return cupy.array(data)

    def __str__(self):
        return f"<kraft.Tensor ({str(self.data)}), device={self.device}, grad_fn={self.grad_fn}>"
