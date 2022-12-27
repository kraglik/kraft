import kraft

from kraft.autograd import Function
from kraft.autograd.utils import match_shape


class Mean(Function):
    def __init__(self, *new_shape):
        self.new_shape = new_shape

    @staticmethod
    def forward(ctx, var, axis, keep_dims):
        np = kraft.get_backend(var)

        return kraft.Variable(
            data=np.mean(
                var.data,
                axis=axis,
                dtype=var.data.dtype,
                keepdims=keep_dims
            ),
            device=var.device,
            dtype=var.data.dtype,
            requires_grad=var.requires_grad,
        )

    @staticmethod
    def backward(ctx, grad):
        var, axis, keep_dims = ctx.inputs
        np = kraft.get_backend(grad)

        g, n = match_shape(grad, np.shape(var.data), axis, keep_dims)

        return g / n


def mean(variable, axis=None, keep_dims=False):
    return Mean()(variable, axis, keep_dims)
