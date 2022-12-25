import kraft
from kraft.autograd import Function


class Neg(Function):
    @staticmethod
    def forward(ctx, var):
        return kraft.Variable(-var.data, device=var.device)

    @staticmethod
    def backward(ctx, grad):
        return -grad


def neg(var):
    return Neg()(var)
