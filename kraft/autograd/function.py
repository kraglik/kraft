from abc import ABC, abstractmethod
from typing import Any
from .tensor import Tensor


class FunctionContext(ABC):
    @abstractmethod
    def save_for_backward(self, *tensors: Tensor) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_saved_tensors(self) -> list[Tensor]:
        raise NotImplementedError


class _ComputationalGraphNode:
    def __init__(self, inputs: Any, outputs: Any, function: 'Function') -> None:
        self._input = inputs
        self._output = outputs
        self._function = function

    @property
    def inputs(self) -> Any:
        return self._input

    @property
    def outputs(self) -> Any:
        return self._output

    @property
    def function(self) -> 'Function':
        return self._function


class Function(ABC):
    def __call__(self, *args: Any):
        pass

    @staticmethod
    @abstractmethod
    def forward(ctx: Any, *args: Any) -> Any:
        """

        :param ctx: Function context
        :param args: Any variables needed to compute function output
        :return:
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        """

        :param ctx:
        :param grad_outputs:
        :return:
        """
        raise NotImplementedError

