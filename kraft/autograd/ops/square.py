import kraft

from kraft.autograd import Function
from kraft.autograd.utils import match_shape


class Square(Function):
    def __init__(self, *new_shape):
        self.new_shape = new_shape

    @staticmethod
    def forward(ctx, var):
        np = kraft.get_backend(var)

        return kraft.Variable(
            data=np.square(var.data),
            device=var.device,
            dtype=var.data.dtype,
            requires_grad=var.requires_grad,
        )

    @staticmethod
    def backward(ctx, grad):
        var, = ctx.inputs
        np = kraft.get_backend(grad)

        return 2 * grad * var.data


def square(variable):
    return Square()(variable)
