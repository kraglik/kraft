import kraft
from kraft.autograd import Function


class Add(Function):
    @staticmethod
    def forward(ctx, left, right):
        result = kraft.Variable(left.data + right.data, device=right.device)
        return result

    @staticmethod
    def backward(ctx, grad):
        return grad, grad


def add(left, right):
    return Add()(left, right)
