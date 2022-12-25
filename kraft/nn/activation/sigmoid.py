from kraft.nn import Module


class Sigmoid(Module):
    def forward(self, x):
        return 1.0 / (1.0 + (-x).exp())


def sigmoid(x):
    return Sigmoid()(x)
