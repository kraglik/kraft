import kraft
from kraft.autograd import Function
from kraft.autograd.utils import match_shape


class Min(Function):
    @staticmethod
    def forward(ctx, var, axis, keep_dims):
        np = kraft.get_backend(var)

        result = kraft.Variable(
            np.min(var.data, axis, keep_dims),
            device=var.device,
            dtype=var.data.dtype,
            requires_grad=var.requires_grad,
        )
        ctx.save_for_backward(result)

        return result

    @staticmethod
    def backward(ctx, grad):
        var, axis, keep_dims = ctx.inputs
        result = ctx.saved_tensors

        return min_max_grad(grad, result.data, var.data, axis, keep_dims)


class Max(Function):
    @staticmethod
    def forward(ctx, var, axis, keep_dims):
        np = kraft.get_backend(var)

        result = kraft.Variable(
            np.max(var.data, axis, keep_dims),
            device=var.device,
            dtype=var.data.dtype,
            requires_grad=var.requires_grad,
        )
        ctx.save_for_backward(result)

        return result

    @staticmethod
    def backward(ctx, grad):
        var, axis, keep_dims = ctx.inputs
        result = ctx.saved_tensors

        return min_max_grad(grad, result.data, var.data, axis, keep_dims)


def min_(var, axis=None, keep_dims=False):
    return Min()(var, axis, keep_dims)


def max_(var, axis=None, keep_dims=False):
    return Max()(var, axis, keep_dims)


def min_max_grad(x, result, inputs, axis, keep_dims):
    np = kraft.get_backend(x)

    reps, _ = match_shape(x, np.shape(x), axis, keep_dims)
    argmax = inputs == match_shape(result, inputs.shape, axis, keep_dims)[0]
    return reps * argmax / np.sum(argmax, axis=axis, keepdims=True)
