import kraft

from kraft.autograd import Function


class Sqrt(Function):
    def __init__(self, *new_shape):
        self.new_shape = new_shape

    @staticmethod
    def forward(ctx, var):
        np = kraft.get_backend(var)

        return kraft.Variable(
            data=np.sqrt(var.data),
            device=var.device,
            dtype=var.data.dtype,
            requires_grad=var.requires_grad,
        )

    @staticmethod
    def backward(ctx, grad):
        var, = ctx.inputs

        return 0.5 * grad * (var.data ** -0.5)


def sqrt(variable):
    return Sqrt()(variable)
