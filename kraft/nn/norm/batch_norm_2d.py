import kraft

from kraft.nn.module import Module, Parameter


class BatchNorm2d(Module):
    def __init__(self, channels: int, momentum=0.1, epsilon=1e-8):
        super().__init__()

        input_shape = (1, channels, 1, 1)

        self.gamma = Parameter(kraft.randn(input_shape))
        self.beta = Parameter(kraft.randn(input_shape))

        self.u_avg = 0.0
        self.std_avg = 0.0

        self.epsilon = epsilon
        self.momentum = momentum

    def forward(self, var):
        if self.training:
            mean = var.mean(axis=0)
            standard_deviation = (var - mean).square().mean(axis=0)
            self.u_avg = self.momentum * self.u_avg + (1 - self.momentum) * mean.data
            self.std_avg = self.momentum * self.std_avg + (1 - self.momentum) * standard_deviation.data

        mean = kraft.Variable(self.u_avg, device=var.device)
        standard_deviation = self.std_avg

        standard_deviation_inverse_sqrt = kraft.Variable(
            1 / (standard_deviation + self.epsilon),
            device=var.device,
        ).sqrt()

        var = (var - mean) * standard_deviation_inverse_sqrt

        return var * self.gamma + self.beta
