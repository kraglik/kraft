import kraft
from kraft.autograd import Function


class Sub(Function):
    @staticmethod
    def forward(ctx, left, right):
        return kraft.Variable(left.data - right.data, device=right.device)

    @staticmethod
    def backward(ctx, grad):
        return grad, -grad


class SubVarFloat(Function):
    @staticmethod
    def forward(ctx, left, right):
        return kraft.Variable(left.data - right, device=left.device)

    @staticmethod
    def backward(ctx, grad):
        return grad


class SubFloatVar(Function):
    @staticmethod
    def forward(ctx, left, right):
        return kraft.Variable(left - right.data, device=right.device)

    @staticmethod
    def backward(ctx, grad):
        return -grad


def sub(left, right):
    return Sub()(left, right)


def sub_var_float(left, right):
    return SubVarFloat()(left, right)


def sub_float_var(left, right):
    return SubFloatVar()(left, right)
