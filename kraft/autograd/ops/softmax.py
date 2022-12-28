import kraft
from kraft.autograd import Function


class Softmax(Function):
    @staticmethod
    def forward(ctx, var, axis):
        xp = kraft.get_backend(var)

        x = var.data

        y = x - x.max(axis=axis, keepdims=True)
        y = xp.exp(y)
        y /= y.sum(axis=axis, keepdims=True)

        ctx.save_for_backward(y)

        return kraft.Variable(
            data=y,
            device=var.device,
            requires_grad=var.requires_grad,
        )

    @staticmethod
    def backward(ctx, gy):
        _, axis = ctx.inputs
        y = ctx.saved_tensors

        gx = y * gy

        sumdx = gx.sum(axis=axis, keepdims=True)
        gx -= y * sumdx

        return gx


def softmax(var, axis=-1):
    return Softmax()(var, axis)
