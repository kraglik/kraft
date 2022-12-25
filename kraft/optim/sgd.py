from .optimizer import Optimizer
from kraft import get_backend


class SGD(Optimizer):
    def __init__(self, parameters, lr=1e-2, nesterov=False, momentum=0.0):
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
        if self._nesterov:
            for parameter in self._parameters:
                parameter.data -= self._cache[parameter] * self._momentum * self._lr

            for parameter in self._parameters:
                dw = parameter.grad
                self._cache[parameter] *= self._momentum
                self._cache[parameter] += dw * (1.0 - self._momentum)
                parameter.data -= dw * (1.0 - self._momentum) * self._lr

        else:
            for parameter in self._parameters:
                dw = parameter.grad
                self._cache[parameter] *= self._momentum
                self._cache[parameter] += dw * (1.0 - self._momentum)
                parameter.data -= self._cache[parameter] * self._lr
