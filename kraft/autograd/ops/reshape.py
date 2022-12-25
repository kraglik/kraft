import kraft

from kraft.autograd import Function


class Reshape(Function):
    def __init__(self, *new_shape):
        self.new_shape = new_shape

    @staticmethod
    def forward(ctx, var, new_shape):
        np = kraft.get_backend(var)

        ctx.save_for_backward(var.shape, new_shape)
        return np.reshape(var, new_shape)

    @staticmethod
    def backward(self, ctx, grad):
        old_shape, new_shape = ctx.saved_tensors
        np = kraft.get_backend(grad)

        if len(new_shape) + 1 == len(grad.shape):
            old_shape = (grad.shape[0], *old_shape)

        return np.reshape(grad, old_shape)


def reshape(variable, new_shape):
    return Reshape()(variable, new_shape)
