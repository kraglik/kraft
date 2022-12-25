from .optimizer import Optimizer
from kraft.device.utils import get_backend


class Adam(Optimizer):

    def __init__(self, parameters, lr=0.001, b1=0.9, b2=0.999, eps=10 ** -8):
        super(Adam, self).__init__(parameters)
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.eps = eps
        self._iteration = 0

        self._cache_m = {
            parameter: get_backend(parameter).zeros(parameter.data.shape)
            for parameter in self._parameters
        }
        self._cache_v = {
            parameter: get_backend(parameter).zeros(parameter.data.shape)
            for parameter in self._parameters
        }

    def zero_grad(self) -> None:
        for parameter in self._parameters:
            parameter.zero_grad()

    def step(self, retain_grad=False) -> None:
        for parameter in self._parameters:
            g = parameter.grad
            np = get_backend(parameter)

            self._cache_m[parameter] = (1 - self.b1) * g + self.b1 * self._cache_m[parameter]
            self._cache_v[parameter] = (1 - self.b2) * (g ** 2) + self.b2 * self._cache_v[parameter]
            mhat = self._cache_m[parameter] / (1 - self.b1 ** (self._iteration + 1))
            vhat = self._cache_v[parameter] / (1 - self.b2 ** (self._iteration + 1))
            parameter.data -= self.lr * mhat / (np.sqrt(vhat) + self.eps)

        self._iteration += 1
