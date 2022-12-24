import numpy as np

from typing import Type
from numpy import single

from kraft.autograd import Tensor


def random(*shape: int, dtype: Type[single] = np.float32, requires_grad: bool = True) -> Tensor:
    raise NotImplementedError


def randn(*shape: int, dtype: Type[single] = np.float32, requires_grad: bool = True) -> Tensor:
    raise NotImplementedError


def zeros(*shape: int, dtype: Type[single] = np.float32, requires_grad: bool = True) -> Tensor:
    raise NotImplementedError


def zeros_like(tensor: Tensor, requires_grad: bool = True) -> Tensor:
    raise NotImplementedError

