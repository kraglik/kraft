from typing import Type, Optional
from numpy import single
from abc import ABC, abstractmethod

import numpy as np

from kraft.device import Device


class _ComputationalGraphNode:
    def __init__(self, input: 'Tensor', output: 'Tensor', function: 'Function') -> None:
        self._input = input
        self._output = output
        self._function = function

    @property
    def input(self) -> 'Tensor':
        return self._input

    @property
    def output(self) -> 'Tensor':
        return self._output

    @property
    def function(self) -> 'Function':
        return self._function


class Function(ABC):
    @abstractmethod
    def forward(self, *args, **kwargs) -> 'Tensor':
        raise NotImplementedError

    @abstractmethod
    def backward(self, gradient: 'Tensor') -> 'Tensor':
        raise NotImplementedError


class Tensor(ABC):
    def __init__(self):
        self._parent_node: _ComputationalGraphNode | None = None

    @abstractmethod
    @property
    def device(self) -> Device:
        raise NotImplementedError

    @abstractmethod
    @property
    def grad(self) -> 'Tensor | None':
        raise NotImplementedError

    def backward(self) -> None:
        pass

    @abstractmethod
    def _set_parent_node(self, node: _ComputationalGraphNode) -> None:
        self._parent_node = node

    @abstractmethod
    def to(self, device: Device) -> 'Tensor':
        raise NotImplementedError

    @abstractmethod
    def to_cpu(self) -> 'Tensor':
        raise NotImplementedError

    @abstractmethod
    def to_gpu(self, index: Optional[int] = None) -> 'Tensor':
        raise NotImplementedError

    @abstractmethod
    def numpy(self) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def zero_grad(self) -> None:
        raise NotImplementedError

    @property
    @abstractmethod
    def dtype(self) -> Type[single]:
        raise NotImplementedError

    @property
    @abstractmethod
    def shape(self) -> list[int]:
        raise NotImplementedError

    @abstractmethod
    def scalar(self) -> float:
        raise NotImplementedError

    @abstractmethod
    def neg(self) -> 'Tensor':
        raise NotImplementedError

    @abstractmethod
    def __abs__(self) -> 'Tensor':
        raise NotImplementedError

    @abstractmethod
    def __add__(self, t):
        raise NotImplementedError

    @abstractmethod
    def __radd__(self, t):
        raise NotImplementedError

    @abstractmethod
    def __sub__(self, t):
        raise NotImplementedError

    @abstractmethod
    def __rsub__(self, t):
        raise NotImplementedError

    @abstractmethod
    def __mul__(self, t):
        raise NotImplementedError

    @abstractmethod
    def __rmul__(self, t):
        raise NotImplementedError

    @abstractmethod
    def __neg__(self):
        raise NotImplementedError

    @abstractmethod
    def __truediv__(self, t):
        raise NotImplementedError

    @abstractmethod
    def __rtruediv__(self, t):
        raise NotImplementedError

    @abstractmethod
    def __pow__(self, t):
        raise NotImplementedError

    @abstractmethod
    def __rpow__(self, t):
        raise NotImplementedError

    @abstractmethod
    def __isub__(self, t):
        raise NotImplementedError

    @abstractmethod
    def __eq__(self, other):
        raise NotImplementedError

    @abstractmethod
    def __ne__(self, other):
        raise NotImplementedError

    @abstractmethod
    def __lt__(self, other):
        raise NotImplementedError

    @abstractmethod
    def __gt__(self, other):
        raise NotImplementedError

    @abstractmethod
    def __le__(self, other):
        raise NotImplementedError

    @abstractmethod
    def __ge__(self, other):
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, item):
        raise NotImplementedError

    @abstractmethod
    def __setitem__(self, key, value):
        raise NotImplementedError
