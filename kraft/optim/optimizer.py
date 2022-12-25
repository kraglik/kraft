from abc import ABC, abstractmethod
from kraft.autograd import Variable


class Optimizer(ABC):
    def __init__(
        self,
        parameters: list[Variable],
    ) -> None:
        self._parameters = parameters

    @abstractmethod
    def step(self, retain_grad=False) -> None:
        raise NotImplementedError

    @abstractmethod
    def zero_grad(self) -> None:
        raise NotImplementedError
