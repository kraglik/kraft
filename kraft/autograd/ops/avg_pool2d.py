import numpy

import kraft
from kraft.autograd import Function
from kraft.autograd.ops.conv2d import col2im
from kraft.autograd.ops.utils.conv import im2col_array, pair, col2im_array
from kraft.autograd.utils import broadcast_to


class AvgPool2d(Function):
    @staticmethod
    def forward(ctx, x, kernel_size, stride, pad):
        col = im2col_array(x.data, kernel_size, stride, pad,
                           to_matrix=False)
        y = col.mean(axis=(2, 3))
        return kraft.Variable(y, requires_grad=x.requires_grad, dtype=x.dtype, device=x.device)

    @staticmethod
    def backward(ctx, grad):
        image, kernel_size, stride, pad = ctx.inputs

        xp = kraft.get_backend(image)

        x = image.data

        N, C, H, W = grad.shape
        KH, KW = pair(kernel_size)

        grad /= KH * KW

        g = xp.broadcast_to(
            grad.reshape(-1),
            (
                KH,
                KW,
                numpy.prod(grad.shape)
            )
        )
        g = xp.reshape(g, newshape=(KH, KW, N, C, H, W))
        g = xp.transpose(g, axes=(2, 3, 0, 1, 4, 5))

        return col2im_array(g, x.shape, pair(kernel_size), stride, pad, to_matrix=False)


def avg_pool2d(x, kernel_size, stride=1, pad=0):
    return AvgPool2d()(x, kernel_size, stride, pad)
