from abc import ABC, abstractmethod

from .module import Parameter


class Regularizer(ABC):
    @abstractmethod
    def get_loss(self, parameters):
        raise NotImplementedError
