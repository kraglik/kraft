import kraft
from kraft.autograd import Function
from kraft.autograd.ops.utils.sum_to import array_sum_to
from kraft.autograd.utils import broadcast_to


class SumTo(Function):
    @staticmethod
    def forward(ctx, x, shape):
        return kraft.Variable(
            data=array_sum_to(x.data, shape),
            device=x.device,
            dtype=x.dtype,
            requires_grad=x.requires_grad,
        )

    @staticmethod
    def backward(ctx, gy):
        x, shape = ctx.inputs

        gx = broadcast_to(x.shape, gy)

        return gx


def sum_to(x, shape):
    return SumTo()(x, shape)
