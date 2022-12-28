import kraft
from kraft.autograd import Function
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

        return (
            broadcast(
                input_grad=grad,
                target_grad=left.data,
            ),
            broadcast(
                input_grad=grad,
                target_grad=right.data,
            )
        )


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
        return broadcast(input_grad=grad, target_grad=ctx.inputs[0].data)


def add(left, right):
    return Add()(left, right)


def add_var_float(left, right):
    return AddVarFloat()(left, right)
