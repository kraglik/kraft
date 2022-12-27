import kraft
from kraft.autograd import Function
from kraft.autograd.ops.utils.conv import im2col_array, col2im_array, pair


class MaxPool2d(Function):
    @staticmethod
    def forward(ctx, x, kernel_size, stride, pad):
        result, indices = _max_pool(x.data, kernel_size, stride, pad)

        ctx.save_for_backward(indices)

        return kraft.Variable(
            result,
            requires_grad=x.requires_grad,
            device=x.device,
            dtype=x.dtype
        )

    @staticmethod
    def backward(ctx, grad):
        x, kernel_size, stride, pad = ctx.inputs
        indices, = ctx.saved_tensors

        return _max_pool_grad(grad, x.data, kernel_size, stride, pad, indices)


def max_pool2d(x, kernel_size, stride=1, pad=0):
    return MaxPool2d()(x, kernel_size, stride, pad)


def _max_pool(x, kernel, stride, padding):
    col = im2col_array(x, kernel, stride, padding,to_matrix=False)
    N, C, KH, KW, OH, OW = col.shape
    col = col.reshape(N, C, KH * KW, OH, OW)
    indexes = col.argmax(axis=2)
    data = col.max(axis=2)
    return data, indexes


def _max_pool_grad(grad, x, kernel, stride, padding, indexes):
    np = kraft.get_backend(x)

    if not isinstance(kernel, tuple):
        kernel = pair(kernel)

    N, C, OH, OW = grad.shape
    N, C, H, W = x.shape
    KH, KW = kernel
    grad_flatten = np.zeros((N * C * OH * OW * KH * KW), dtype=np.float)
    indexes = (indexes.ravel() + np.arange(0, indexes.size * KH * KW, KH * KW))
    grad_flatten[indexes] = grad.ravel()
    g = grad_flatten.reshape(N, C, OH, OW, KH, KW)
    g = np.swapaxes(g, 2, 4)
    g = np.swapaxes(g, 3, 5)
    return col2im_array(g, (N, C, H, W), kernel, stride, padding, to_matrix=False)
