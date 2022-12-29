from kraft.nn import Regularizer


class L1Regularizer(Regularizer):
    def __init__(self, alpha: float = 1e-2, reduction="mean"):
        self._alpha = alpha
        self._reduction = reduction

    def get_loss(self, parameters):
        addition = 0

        for parameter in parameters:
            if self._reduction == "mean":
                addition = addition + parameter.abs().mean()
            else:
                addition = addition + parameter.abs().sum()

        if self._reduction == "mean":
            return self._alpha * (addition / len(parameters))

        return self._alpha * addition

