from abc import ABC, abstractmethod

from .module import Parameter


class Regularizer(ABC):
    def __init__(self, parameters):
        self._parameters = parameters

    @abstractmethod
    def get_loss(self):
        raise NotImplementedError
