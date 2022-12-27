import kraft
from kraft.autograd import Function


class Flatten(Function):
    @staticmethod
    def forward(ctx, var):
        np = kraft.get_backend(var)
        ctx.save_for_backward(var.shape)

        return kraft.Variable(
            data=np.reshape(var.data, (var.shape[0], -1)),
            device=var.device,
            dtype=var.data.dtype,
            requires_grad=var.requires_grad
        )

    @staticmethod
    def backward(ctx, grad):
        np = kraft.get_backend(grad)
        old_shape, = ctx.saved_tensors

        return np.reshape(grad, old_shape)


def flatten(var):
    return Flatten()(var)
