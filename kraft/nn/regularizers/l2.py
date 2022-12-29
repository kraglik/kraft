from kraft.nn import Regularizer


class L2Regularizer(Regularizer):
    def __init__(self, alpha: float = 1e-2, reduction="mean"):
        self._alpha = alpha
        self._reduction = reduction

    def get_loss(self, parameters):
        addition = 0

        for parameter in parameters:
            p = parameter.square().flatten()

            if self._reduction == "mean":
                addition = addition + p.mean() / (len(parameters))
            else:
                addition = addition + p.sum()

        return self._alpha * addition

