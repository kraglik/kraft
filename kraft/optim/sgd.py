from .optimizer import Optimizer
from kraft import get_backend


class SGD(Optimizer):
    def __init__(
        self,
        parameters,
        lr=1e-2,
        momentum=0.0,
        nesterov=False,
    ):
        super().__init__(parameters)

        self._lr = lr
        self._momentum = momentum
        self._nesterov = nesterov
        self._cache = {
            parameter: get_backend(parameter).zeros(parameter.data.shape)
            for parameter in self._parameters
        }

    def zero_grad(self):
        for param in self._parameters:
            param.zero_grad()

    def step(self, retain_graph=False):
        for parameter in self._parameters:
            if not parameter.requires_grad:
                continue

            # A trivial case
            if self._momentum == 0.0:
                dw = parameter.grad
                parameter.data -= dw * self._lr

            # A more sophisticated case with momentum
            else:
                dw = parameter.grad * (1.0 - self._momentum) * self._lr
                cache = self._cache[parameter] * self._momentum
                update = dw + cache

                self._cache[parameter] = update
                parameter.data -= update

                if self._nesterov:
                    parameter.data -= self._momentum * update
