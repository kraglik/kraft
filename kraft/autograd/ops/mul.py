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


class MulVarFloat(Function):
    @staticmethod
    def forward(ctx, left, right):
        return kraft.Variable(left.data * right, device=left.device)

    @staticmethod
    def backward(ctx, grad):
        _, right = ctx.inputs

        return right * grad


def mul(left, right):
    return Mul()(left, right)


def mul_var_float(left, right):
    return MulVarFloat()(left, right)
