import kraft
from kraft.autograd import Function


class Div(Function):
    @staticmethod
    def forward(ctx, left, right):
        output = kraft.Variable(
            left.data / right.data,
            device=right.device,
            requires_grad=left.requires_grad or right.requires_grad,
        )
        ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx, grad):
        left, right = ctx.inputs
        output, = ctx.saved_tensors

        left_grad = grad / right.data
        right_grad = -grad * output.data ** 2

        return left_grad, right_grad


class DivVarFloat(Function):
    @staticmethod
    def forward(ctx, left, right):
        output = kraft.Variable(left.data / right, device=right.device)
        ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx, grad):
        left, right = ctx.inputs
        output, = ctx.saved_tensors

        left_grad = grad / right.data

        return left_grad


class DivFloatVar(Function):
    @staticmethod
    def forward(ctx, left, right):
        output = kraft.Variable(left / right.data, device=right.device)
        ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx, grad):
        left, right = ctx.inputs
        output, = ctx.saved_tensors

        right_grad = -grad * output.data ** 2

        return right_grad


def div(left, right):
    return Div()(left, right)


def div_var_float(left, right):
    return DivVarFloat()(left, right)


def div_float_var(left, right):
    return DivFloatVar()(left, right)
