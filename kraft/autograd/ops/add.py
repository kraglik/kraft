import kraft
from kraft.autograd import Function


class Add(Function):
    @staticmethod
    def forward(ctx, left, right):
        result = kraft.Variable(
            left.data + right.data,
            device=right.device,
            requires_grad=left.requires_grad or right.requires_grad,
        )
        return result

    @staticmethod
    def backward(ctx, grad):
        return grad, grad


class AddVarFloat(Function):
    @staticmethod
    def forward(ctx, left, right):
        result = kraft.Variable(
            left.data + right,
            device=left.device,
            requires_grad=left.requires_grad
        )
        return result

    @staticmethod
    def backward(ctx, grad):
        return grad


def add(left, right):
    return Add()(left, right)


def add_var_float(left, right):
    return AddVarFloat()(left, right)
