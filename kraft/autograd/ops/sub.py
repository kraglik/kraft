import kraft
from kraft.autograd import Function


class Sub(Function):
    @staticmethod
    def forward(ctx, left, right):
        return kraft.Variable(left.data - right.data, device=right.device)

    @staticmethod
    def backward(ctx, grad):
        return grad, -grad


def sub(left, right):
    return Sub()(left, right)
