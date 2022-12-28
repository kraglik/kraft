from kraft.nn import Regularizer


class L2Regularizer(Regularizer):
    def __init__(self, parameters, alpha: float = 1e-2, reduction="mean"):
        super().__init__(parameters)
        self._alpha = alpha
        self._reduction = reduction

    def get_loss(self):
        addition = 0

        for parameter in self._parameters:
            p = parameter.square().flatten()

            if self._reduction == "mean":
                addition = addition + p.mean() / (len(self._parameters))
            else:
                addition = addition + p.sum()

        return self._alpha * addition

