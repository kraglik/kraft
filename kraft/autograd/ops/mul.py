import kraft
from kraft.autograd import Function


class Mul(Function):
    @staticmethod
    def forward(ctx, left, right):
        return kraft.Variable(left.data * right.data, device=right.device)

    @staticmethod
    def backward(ctx, grad):
        left, right = ctx.inputs

        return right.data * grad, left.data * grad


def mul(left, right):
    return Mul()(left, right)
