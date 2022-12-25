import kraft
from kraft.autograd import Function


class Flatten(Function):
    @staticmethod
    def forward(ctx, var):
        np = kraft.get_backend(var)
        ctx.save_for_backward(var.shape)

        return np.reshape(var, (var.shape[0], -1))

    @staticmethod
    def backward(ctx, grad):
        np = kraft.get_backend(grad)
        old_shape, = ctx.data_for_back

        return np.reshape(grad, old_shape)


def flatten(var):
    return Flatten()(var)
