from abc import ABC, abstractmethod
from kraft.autograd import Tensor


class Optimizer(ABC):
    def __init__(
        self,
        parameters: list[Tensor],
    ) -> None:
        self._parameters = parameters

    @abstractmethod
    def step(self, preserve_graph=False) -> None:
        raise NotImplementedError

    @abstractmethod
    def zero_grad(self) -> None:
        raise NotImplementedError
