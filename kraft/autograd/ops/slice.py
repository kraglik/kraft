import kraft
from kraft.autograd import Function


class Slice(Function):
    @staticmethod
    def forward(ctx, var, indices):
        data = var.data[indices]

        return kraft.Variable(
            data=data,
            device=var.device,
            dtype=var.dtype,
            requires_grad=var.requires_grad,
        )

    @staticmethod
    def backward(ctx, grad):
        var, indices = ctx.inputs
        np = kraft.get_backend(var)

        grad_ = np.zeros_like(var.data)
        grad_[indices] = grad

        return grad_


def slice_(var, indices):
    return Slice()(var, indices)
