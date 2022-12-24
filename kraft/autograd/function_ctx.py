from abc import ABC, abstractmethod

from .tensor import Tensor


class FunctionContext(ABC):
    @abstractmethod
    def save_for_backward(self, *tensors: 'Tensor') -> None:
        raise NotImplementedError

    @abstractmethod
    def get_saved_tensors(self) -> list['Tensor']:
        raise NotImplementedError
