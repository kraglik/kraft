import kraft
from kraft.autograd import Function
from kraft.autograd.ops.conv2d import col2im
from kraft.autograd.ops.utils.conv import im2col_array, pair
from kraft.autograd.utils import broadcast_to


class AvgPool2d(Function):
    @staticmethod
    def forward(ctx, x, kernel_size, stride, pad):
        col = im2col_array(x.data, kernel_size, stride, pad,
                           to_matrix=False)
        y = col.mean(axis=(2, 3))
        return kraft.Variable(y, requires_grad=x.requires_grad, dtype=x.dtype, device=x.device)

    @staticmethod
    def backward(ctx, gy):
        x, kernel_size, stride, pad = ctx.inputs

        N, C, OH, OW = gy.shape
        KW, KH = pair(kernel_size)
        gy /= (KW*KH)
        gcol = broadcast_to((KH, KW, N*C*OH*OW), gy.reshape(-1))
        gcol = gcol.reshape(KH, KW, N, C, OH, OW).transpose(2, 3, 0, 1, 4, 5)
        gx = col2im(gcol, x.shape, kernel_size, stride, pad, to_matrix=False)
        return gx


def avg_pool2d(x, kernel_size, stride=1, pad=0):
    return AvgPool2d()(x, kernel_size, stride, pad)
