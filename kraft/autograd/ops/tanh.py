import kraft
from kraft.autograd import Function


class Tanh(Function):
    @staticmethod
    def forward(ctx, var):
        np = kraft.get_backend(var)

        result = kraft.Variable(
            np.tanh(var),
            device=var.device,
            requires_grad=var.requires_grad,
            dtype=var.data.dtype,
        )
        return result

    @staticmethod
    def backward(ctx, grad):
        var, = ctx.inputs
        np = kraft.get_backend(var)

        return grad / np.cosh(var.data) ** 2


def tanh(var):
    return Tanh()(var)
