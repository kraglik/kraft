from kraft.nn import Module


class ReLU(Module):
    def forward(self, x):
        return x * (x >= 0.0)


def relu(x):
    return ReLU()(x)
