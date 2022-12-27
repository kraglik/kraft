from kraft.nn import Module


class Tanh(Module):
    def forward(self, x):
        return x.tanh()


def tanh(x):
    return x.tanh()
