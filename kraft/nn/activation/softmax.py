from kraft.nn import Module


class Softmax(Module):
    def forward(self, var):
        y = (var - var.max(axis=-1, keep_dims=True)).exp()
        return y / y.sum(axis=-1, keep_dims=True)


def softmax(var):
    return Softmax()(var)
