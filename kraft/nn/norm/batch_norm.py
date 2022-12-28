import math

import kraft

from kraft.nn.module import Module, Parameter
from kraft.autograd import Variable


class BatchNorm(Module):
    def __init__(self, input_shape, momentum=0.1, epsilon=1e-8):
        super().__init__()
        self.beta = Parameter(kraft.randn(input_shape))
        self.gamma = Parameter(kraft.randn(input_shape))
        self.u_avg = 0
        self.std_avg = 0
        self.epsilon = epsilon
        self.momentum = momentum

    def forward(self, x):
        if self.training:
            mean = x.mean()
            standard_deviation = (x - mean).square().mean()
            self.u_avg = (self.momentum * self.u_avg + (1 - self.momentum) * mean.data).item()
            self.std_avg = (self.momentum * self.std_avg + (1 - self.momentum) * standard_deviation.data).item()
            standard_deviation_root = (standard_deviation + self.epsilon).sqrt()
        else:
            mean = self.u_avg
            standard_deviation = self.std_avg
            standard_deviation_root = math.sqrt(standard_deviation + self.epsilon)

        x = (x - mean) / standard_deviation_root
        return x * self.gamma + self.beta
