import kraft
from kraft.autograd import Function
from kraft.autograd.ops.utils.sum_to import array_sum_to
from kraft.autograd.utils import broadcast


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
        left, right = ctx.inputs
        left_grad, right_grad = grad, grad

        if left.shape != right.shape:  # for broadcaset
            left_grad = array_sum_to(left_grad, left.shape)
            right_grad = array_sum_to(right_grad, right.shape)

        return left_grad, right_grad


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
        var, _ = ctx.inputs

        if var.shape != grad.shape:
            grad = array_sum_to(grad, var.shape)

        return grad


def add(left, right):
    return Add()(left, right)


def add_var_float(left, right):
    return AddVarFloat()(left, right)
