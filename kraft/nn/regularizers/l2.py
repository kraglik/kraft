from kraft.nn import Regularizer


class L2Regularizer(Regularizer):
    def __init__(self, parameters, alpha: float = 1e-2):
        super().__init__(parameters)
        self._alpha = alpha

    def add_to_loss(self, loss):
        addition = 0

        for parameter in self._parameters:
            addition = addition + parameter.square().mean()

        return loss + self._alpha * addition

