import kraft
from kraft.autograd import Function


class Div(Function):
    @staticmethod
    def forward(ctx, left, right):
        output = kraft.Variable(left.data / right.data, device=right.device)
        ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx, grad):
        left, right = ctx.inputs
        output, = ctx.saved_tensors

        left_grad = grad / right.data
        right_grad = -grad * output.data ** 2

        return left_grad, right_grad


def div(left, right):
    return Div()(left, right)
