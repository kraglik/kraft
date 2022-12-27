import kraft

from kraft.autograd import Function


class Log(Function):
    def __init__(self, *new_shape):
        self.new_shape = new_shape

    @staticmethod
    def forward(ctx, var):
        np = kraft.get_backend(var)

        return kraft.Variable(
            data=np.log(var.data),
            device=var.device,
            dtype=var.data.dtype,
            requires_grad=var.requires_grad,
        )

    @staticmethod
    def backward(ctx, grad):
        var, = ctx.inputs

        return grad / var.data


def log(variable):
    return Log()(variable)
