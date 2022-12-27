import kraft

from kraft.autograd import Function


class Abs(Function):
    def __init__(self, *new_shape):
        self.new_shape = new_shape

    @staticmethod
    def forward(ctx, var):
        np = kraft.get_backend(var)

        return kraft.Variable(
            data=np.abs(var.data),
            device=var.device,
            dtype=var.dtype,
            requires_grad=var.requires_grad,
        )

    @staticmethod
    def backward(ctx, grad):
        var, = ctx.inputs
        np = kraft.get_backend(grad)

        return grad * np.sign(var.data)


def abs_(variable):
    return Abs()(variable)
