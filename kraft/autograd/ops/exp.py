import kraft
from kraft.autograd import Function


class Exp(Function):
    @staticmethod
    def forward(ctx, var):
        np = kraft.get_backend(var)

        output = kraft.Variable(np.exp(var.data), device=var.device)
        ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx, grad):
        output, = ctx.saved_tensors
        return output.data * grad


def exp(var):
    return Exp()(var)
