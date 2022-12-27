from kraft.nn import Module


class Softmax(Module):
    def forward(self, var):
        y = (var - var.max(axis=-1, keepdims=True)).exp()
        return y / y.sum(axis=-1, keepdims=True)


def softmax(var):
    return Softmax()(var)
