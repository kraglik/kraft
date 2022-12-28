import kraft

from kraft.autograd import Function
from kraft.autograd.ops.utils.conv import pair, get_conv_outsize, get_deconv_outsize, col2im_array, im2col_array


class Conv2d(Function):
    @staticmethod
    def forward(ctx, x, W, b, stride, pad):
        xp = kraft.get_backend(x)
        stride, pad = pair(stride), pair(pad)

        KH, KW = W.shape[2:]
        col = im2col_array(x.data, (KH, KW), stride, pad, to_matrix=False)

        y = xp.tensordot(col, W.data, ((1, 2, 3), (1, 2, 3)))

        if b is not None:
            y += b.data

        y = xp.rollaxis(y, 3, 1)

        return kraft.Variable(y, requires_grad=x.requires_grad, device=x.device, dtype=x.dtype)

    @staticmethod
    def backward(ctx, gy):
        x, W, b, stride, pad = ctx.inputs

        stride, pad = pair(stride), pair(pad)

        # ==== gx ====
        gx = Deconv2d.forward(
            None,
            gy,
            W,
            b=None,
            stride=stride,
            pad=pad,
            outsize=(x.shape[2], x.shape[3])
        ).data
        # ==== gW ====
        gW = Conv2DGradW.forward(None, x, gy, W.shape[2:], stride, pad)
        # ==== gb ====
        gb = None

        if b.data is not None:
            gb = gy.sum(axis=(0, 2, 3))

        return gx, gW, gb


def conv2d(x, W, b=None, stride=1, pad=0):
    return Conv2d()(x, W, b, stride, pad)


class Deconv2d(Function):
    @staticmethod
    def forward(ctx, x, W, b, stride, pad, outsize):
        xp = kraft.get_backend(x)

        stride, pad = pair(stride), pair(pad)

        Weight = W
        SH, SW = stride
        PH, PW = pad
        C, OC, KH, KW = Weight.shape
        N, C, H, W = x.shape

        if outsize is None:
            out_h = get_deconv_outsize(H, KH, SH, PH)
            out_w = get_deconv_outsize(W, KW, SW, PW)
        else:
            out_h, out_w = pair(outsize)
        img_shape = (N, OC, out_h, out_w)

        gcol = xp.tensordot(Weight.data, (x.data if isinstance(x, kraft.Variable) else x), (0, 1))
        gcol = xp.rollaxis(gcol, 3)
        y = col2im_array(gcol, img_shape, (KH, KW), stride, pad,
                         to_matrix=False)
        # b, k, h, w
        if b is not None:
            y += b.data.reshape((1, b.size, 1, 1))

        return kraft.Variable(y, requires_grad=Weight.requires_grad, device=Weight.device, dtype=Weight.dtype)

    @staticmethod
    def backward(ctx, gy):
        x, W, b, stride, pad, outsize = ctx.inputs
        stride, pad = pair(stride), pair(pad)

        # ==== gx ====
        gx = conv2d(gy, W, b=None, stride=stride, pad=pad).data
        # ==== gW ====
        gW = Conv2DGradW.forward(None, gy, x, W.shape[2:], stride, pad)
        # ==== gb ====
        gb = None

        if b.data is not None:
            gb = gy.sum(axis=(0, 2, 3))

        return gx, gW, gb


def deconv2d(x, W, b=None, stride=1, pad=0, outsize=None):
    return Deconv2d()(x, W, b, stride, pad, outsize)


class Conv2DGradW(Function):
    @staticmethod
    def forward(ctx, x, gy, kernel_size, stride, pad):
        xp = kraft.get_backend(x)

        col = im2col_array(
            x.data if isinstance(x, kraft.Variable) else x,
            kernel_size,
            stride,
            pad,
            to_matrix=False
        )
        gW = xp.tensordot(gy, col, ((0, 2, 3), (0, 4, 5)))

        if ctx is not None:
            ctx.save_for_backward(gW)

        return gW

    @staticmethod
    def backward(ctx, gys):
        x, gy, kernel_size, stride, pad = ctx.inputs
        gW, = ctx.saved_tensors

        xh, xw = x.shape[2:]
        gx = deconv2d(gy, gW, stride=stride, pad=pad,
                      outsize=(xh, xw))
        ggy = conv2d(x, gW, stride=stride, pad=pad)
        return gx.data, ggy.data


# =============================================================================
#  im2col / col2im
# =============================================================================

class Im2col(Function):
    @staticmethod
    def forward(ctx, x, kernel_size, stride, pad, to_matrix):
        y = im2col_array(x, kernel_size, stride, pad, to_matrix)
        return y

    @staticmethod
    def backward(ctx, gy):
        x, kernel_size, stride, pad, to_matrix = ctx.inputs

        gx = col2im(gy, x.shape, kernel_size, stride, pad, to_matrix)
        return gx


def im2col(x, kernel_size, stride=1, pad=0, to_matrix=True):
    y = Im2col()(x, kernel_size, stride, pad, to_matrix)
    return y


class Col2im(Function):
    @staticmethod
    def forward(ctx, x, input_shape, kernel_size, stride, pad, to_matrix):
        y = col2im_array(
            x,
            input_shape,
            kernel_size,
            stride,
            pad,
            to_matrix
        )
        return y

    @staticmethod
    def backward(ctx, gy):
        _, input_shape, kernel_size, stride, pad, to_matrix = ctx.inputs
        gx = im2col(gy, kernel_size, stride, pad, to_matrix)
        return gx


def col2im(x, input_shape, kernel_size, stride=1, pad=0, to_matrix=True):
    return Col2im()(x, input_shape, kernel_size, stride, pad, to_matrix)
