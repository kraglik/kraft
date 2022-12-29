from .optimizer import Optimizer
from kraft.device.utils import get_backend


class Adam(Optimizer):

    def __init__(self, parameters, lr=0.001, b1=0.99, b2=0.999, eps=10 ** -8):
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
        self._iteration += 1

        for parameter in self._parameters:
            if not parameter.requires_grad:
                continue

            g = parameter.grad
            np = get_backend(parameter)

            self._cache_m[parameter] = (1 - self.b1) * g + self.b1 * self._cache_m[parameter]
            self._cache_v[parameter] = (1 - self.b2) * (g * g) + self.b2 * self._cache_v[parameter]
            mhat = self._cache_m[parameter] / (1 - self.b1 ** self._iteration)
            vhat = self._cache_v[parameter] / (1 - self.b2 ** self._iteration)
            parameter.data -= self.lr * mhat / (np.sqrt(vhat) + self.eps)
