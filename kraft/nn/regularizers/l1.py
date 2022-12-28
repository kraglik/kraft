from kraft.nn import Regularizer


class L1Regularizer(Regularizer):
    def __init__(self, parameters, alpha: float = 1e-2, reduction="mean"):
        super().__init__(parameters)
        self._alpha = alpha
        self._reduction = reduction

    def add_to_loss(self, loss):
        addition = 0

        for parameter in self._parameters:
            if self._reduction == "mean":
                addition = addition + parameter.abs().mean()
            else:
                addition = addition + parameter.abs().sum()

        if self._reduction == "mean":
            return loss + self._alpha * (addition / len(self._parameters))

        return loss + self._alpha * addition

