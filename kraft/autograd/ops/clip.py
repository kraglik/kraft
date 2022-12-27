import kraft
from kraft.autograd import Function


class Clip(Function):
    @staticmethod
    def forward(ctx, var, minimum, maximum):
        np = kraft.get_backend(var)

        result = kraft.Variable(
            np.clip(var.data, minimum, maximum),
            device=var.device,
            dtype=var.data.dtype,
            requires_grad=var.requires_grad,
        )
        return result

    @staticmethod
    def backward(ctx, grad):
        var, minimum, maximum = ctx.inputs

        np = kraft.get_backend(var)

        return grad * np.logical_and(var.data != minimum, var.data != maximum)


def clip(var, minimum, maximum):
    return Clip()(var, minimum, maximum)
