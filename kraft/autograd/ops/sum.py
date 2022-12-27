import kraft
from kraft.autograd import Function
from kraft.autograd.utils import match_shape


class Sum(Function):
    @staticmethod
    def forward(ctx, var, axis, keep_dims):
        np = kraft.get_backend(var)

        return kraft.Variable(np.sum(var.data, axis, keep_dims), device=var.device)

    @staticmethod
    def backward(ctx, grad):
        var, axis, keep_dims = ctx.inputs

        sum_grad = match_shape(grad, var.shape, axis, keep_dims)[0]

        return sum_grad


def sum_var(var, axis=None, keep_dims=False):
    return Sum()(var, axis, keep_dims)
