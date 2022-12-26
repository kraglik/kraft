import kraft
from kraft.autograd import Function
from kraft.autograd.ops.utils.sum_to import array_sum_to
from kraft.autograd.utils import broadcast_to


class SumTo(Function):
    @staticmethod
    def forward(ctx, x, shape):
        y = array_sum_to(x, shape)
        return y

    @staticmethod
    def backward(ctx, gy):
        x, shape = ctx.inputs

        gx = broadcast_to(x.shape, gy)

        return gx


def sum_to(x, shape):
    return SumTo()(x, shape)
