from abc import ABC, abstractmethod
from typing import Any


class Function(ABC):
    """

    """

    @staticmethod
    @abstractmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any) -> Any:
        """

        :param ctx: Function context
        :param args: Any positional variables needed to compute function output
        :param kwargs: Any named variables needed to compute function output
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
