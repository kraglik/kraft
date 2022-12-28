import kraft

from kraft.autograd import Function
from kraft.autograd.ops.utils.conv import pair, get_conv_outsize, get_deconv_outsize, col2im_array, im2col_array


class Conv2d(Function):
    @staticmethod
    def forward(ctx, image, weights, stride, pad):
        np = kraft.get_backend(image)
        stride, pad = pair(stride), pair(pad)

        kernel_height, kernel_width = weights.shape[2:]

        col = im2col_array(image.data, (kernel_height, kernel_width), stride, pad, to_matrix=False)
        data = np.tensordot(col, weights.data, ((1, 2, 3), (1, 2, 3)))
        data = np.rollaxis(data, 3, 1)

        return kraft.Variable(
            data=data,
            requires_grad=image.requires_grad,
            device=image.device,
            dtype=image.dtype
        )

    @staticmethod
    def backward(ctx, grad):
        image, weights, stride, pad = ctx.inputs

        np = kraft.get_backend(image)

        stride, pad = pair(stride), pair(pad)

        image_grad = _deconv(
            grad,
            weights,
            stride=stride,
            pad=pad,
        )

        kernel_height, kernel_width = weights.shape[2:]
        col = im2col_array(image.data, (kernel_height, kernel_width), stride, pad, to_matrix=False)

        weights_grad = np.tensordot(grad, col, ((0, 2, 3), (0, 4, 5)))

        return image_grad, weights_grad


def conv2d(image, weights, stride=1, pad=0):
    return Conv2d()(image, weights, stride, pad)


class Deconv2d(Function):
    @staticmethod
    def forward(ctx, image, weights, bias, stride, pad):
        np = kraft.get_backend(image)

        stride, pad = pair(stride), pair(pad)

        Weight = weights
        stride_height, stride_width = stride
        pad_height, pad_width = pad
        in_channels, out_channels, kernel_height, kernel_width = Weight.shape
        batch_size, in_channels, height, width = image.shape

        out_h = get_deconv_outsize(height, kernel_height, stride_height, pad_height)
        out_w = get_deconv_outsize(weights, kernel_width, stride_width, pad_width)

        img_shape = (batch_size, out_channels, out_h, out_w)

        gcol = np.tensordot(Weight.data, (image.data if isinstance(image, kraft.Variable) else image), (0, 1))
        gcol = np.rollaxis(gcol, 3)
        y = col2im_array(
            gcol,
            img_shape,
            (kernel_height, kernel_width),
            stride,
            pad,
            to_matrix=False
        )

        if bias is not None:
            y += bias.data.reshape((1, bias.size, 1, 1))

        return kraft.Variable(y, requires_grad=Weight.requires_grad, device=Weight.device, dtype=Weight.dtype)

    @staticmethod
    def backward(ctx, grad):
        image, weights, bias, stride, pad, outsize = ctx.inputs
        stride, pad = pair(stride), pair(pad)

        image_grad = conv2d(grad, weights, bias=None, stride=stride, pad=pad).data
        weights_grad = Conv2DGradW.forward(None, grad, image, weights.shape[2:], stride, pad)
        bias_grad = None

        if bias.data is not None:
            bias_grad = grad.sum(axis=(0, 2, 3))

        return image_grad, weights_grad, bias_grad


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


def _deconv(image, weights, stride, pad):
    np = kraft.get_backend(image)

    stride, pad = pair(stride), pair(pad)

    stride_height, stride_width = stride
    pad_height, pad_width = pad
    in_channels, out_channels, kernel_height, kernel_width = weights.shape
    batch_size, in_channels, height, width = image.shape

    out_h = get_deconv_outsize(height, kernel_height, stride_height, pad_height)
    out_w = get_deconv_outsize(width, kernel_width, stride_width, pad_width)

    img_shape = (batch_size, out_channels, out_h, out_w)

    gcol = np.tensordot(weights.data, (image.data if isinstance(image, kraft.Variable) else image), (0, 1))
    gcol = np.rollaxis(gcol, 3)
    y = col2im_array(
        gcol,
        img_shape,
        (kernel_height, kernel_width),
        stride,
        pad,
        to_matrix=False
    )

    return y
