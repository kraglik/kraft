import kraft
from kraft.autograd import Function
from kraft.autograd.ops.utils.conv import im2col_array, col2im_array, pair


class MaxPool2d(Function):
    @staticmethod
    def forward(ctx, x, kernel_size, stride, pad):
        col = im2col_array(x.data, kernel_size, stride, pad, to_matrix=False)

        N, C, KH, KW, OH, OW = col.shape
        col = col.reshape(N, C, KH * KW, OH, OW)
        indexes = col.argmax(axis=2)
        y = col.max(axis=2)

        ctx.save_for_backward(indexes)

        return kraft.Variable(y, requires_grad=x.requires_grad, device=x.device, dtype=x.dtype)

    @staticmethod
    def backward(ctx, gy):
        x, kernel_size, stride, pad = ctx.inputs
        indexes, = ctx.saved_tensors

        result = Pooling2DGrad.forward(
            None,
            kraft.Variable(
                gy,
                requires_grad=x.requires_grad,
                device=x.device,
                dtype=x.dtype,
            ),
            kernel_size,
            stride,
            pad,
            x.shape,
            x.dtype,
            indexes
        )

        return result


class Pooling2DGrad(Function):
    @staticmethod
    def forward(ctx, gy, kernel_size, stride, pad, input_shape, dtype, indexes):
        xp = kraft.get_backend(gy)

        N, C, OH, OW = gy.shape
        N, C, H, W = input_shape
        KH, KW = pair(kernel_size)

        gcol = xp.zeros((N * C * OH * OW * KH * KW), dtype=dtype)

        indexes = (indexes.ravel() + xp.arange(0, indexes.size * KH * KW, KH * KW))

        if ctx is not None:
            ctx.save_for_backward(indexes)

        gcol[indexes] = gy.data.ravel()
        gcol = gcol.reshape(N, C, OH, OW, KH, KW)
        gcol = xp.swapaxes(gcol, 2, 4)
        gcol = xp.swapaxes(gcol, 3, 5)

        gx = col2im_array(gcol, (N, C, H, W), kernel_size, stride, pad, to_matrix=False)
        return gx

    @staticmethod
    def backward(ctx, ggx):
        x, kernel_size, stride, pad, input_shape, dtype, index = ctx.inputs
        indexes, = ctx.saved_tensors

        np = kraft.get_backend(x)

        col = im2col_array(x, kernel_size, stride, pad,
                           to_matrix=False)
        N, C, KH, KW, OH, OW = col.shape
        col = col.reshape(N, C, KH * KW, OH, OW)
        col = col.transpose(0, 1, 3, 4, 2).reshape(-1, KH * KW)
        indexes = indexes.ravel()
        col = col[np.arange(len(indexes)), indexes]

        return col.reshape(N, C, OH, OW)


def max_pool2d(x, kernel_size, stride=1, pad=0):
    return MaxPool2d()(x, kernel_size, stride, pad)
