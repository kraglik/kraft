from abc import ABC, abstractmethod
from typing import Any, Optional, Type

import numpy as np
from numpy import single

from kraft.device import Device


class Tensor(ABC):
    def __init__(self):
        self._parent_node: Any = None

    @property
    def device(self) -> Device:
        raise NotImplementedError

    @property
    def requires_grad(self) -> bool:
        raise NotImplementedError

    @requires_grad.setter
    def requires_grad(self, value) -> None:
        raise NotImplementedError

    @property
    def grad(self) -> 'Tensor | None':
        raise NotImplementedError

    def detach(self) -> 'Tensor':
        raise NotImplementedError

    def _set_parent_node(self, node: Any) -> None:
        self._parent_node = node

    def backward(self) -> None:
        pass

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
    def shape(self) -> list[int]:
        raise NotImplementedError

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
