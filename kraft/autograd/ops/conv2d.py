import numpy as np

import kraft

from kraft.autograd import Function
from kraft.autograd.ops.utils.conv import pair, get_conv_outsize, get_deconv_outsize


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
        gx = deconv2d(
            gy,
            W,
            b=None,
            stride=stride,
            pad=pad,
            outsize=(x.shape[2], x.shape[3])
        ).data
        # ==== gW ====
        gW = Conv2DGradW()(x, gy, W.shape[2:], stride, pad)
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
        gW = Conv2DGradW()(gy, x, W.shape[2:], stride, pad).data
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


def im2col_array(img, kernel_size, stride, pad, to_matrix=True):
    N, C, H, W = img.shape
    KH, KW = pair(kernel_size)
    SH, SW = pair(stride)
    PH, PW = pair(pad)
    OH = get_conv_outsize(H, KH, SH, PH)
    OW = get_conv_outsize(W, KW, SW, PW)

    xp = kraft.get_backend(img)
    if xp != np:
        col = _im2col_gpu(img, kernel_size, stride, pad)
    else:
        img = np.pad(img,
                     ((0, 0), (0, 0), (PH, PH + SH - 1), (PW, PW + SW - 1)),
                     mode='constant', constant_values=(0,))
        col = np.ndarray((N, C, KH, KW, OH, OW), dtype=img.dtype)

        for j in range(KH):
            j_lim = j + SH * OH
            for i in range(KW):
                i_lim = i + SW * OW
                col[:, :, j, i, :, :] = img[:, :, j:j_lim:SH, i:i_lim:SW]

    if to_matrix:
        col = col.transpose((0, 4, 5, 1, 2, 3)).reshape((N * OH * OW, -1))

    return col


def col2im_array(col, img_shape, kernel_size, stride, pad, to_matrix=True):
    N, C, H, W = img_shape
    KH, KW = pair(kernel_size)
    SH, SW = pair(stride)
    PH, PW = pair(pad)
    OH = get_conv_outsize(H, KH, SH, PH)
    OW = get_conv_outsize(W, KW, SW, PW)

    if to_matrix:
        col = col.reshape(N, OH, OW, C, KH, KW).transpose(0, 3, 4, 5, 1, 2)

    xp = kraft.get_backend(col)

    if xp != np:
        img = _col2im_gpu(col, SH, SW, PH, PW, H, W)
        return img
    else:
        img = np.zeros((N, C, H + 2 * PH + SH - 1, W + 2 * PW + SW - 1),
                       dtype=col.dtype)
        for j in range(KH):
            j_lim = j + SH * OH
            for i in range(KW):
                i_lim = i + SW * OW
                img[:, :, j:j_lim:SH, i:i_lim:SW] += col[:, :, j, i, :, :]
        return img[:, :, PH:H + PH, PW:W + PW]


def _im2col_gpu(img, kernel_size, stride, pad):
    """im2col function for GPU.
    This code is ported from Chainer:
    https://github.com/chainer/chainer/blob/v6.4.0/chainer/utils/conv.py
    """
    import cupy

    n, c, h, w = img.shape
    kh, kw = pair(kernel_size)
    sy, sx = pair(stride)
    ph, pw = pair(pad)
    out_h = get_conv_outsize(h, kh, sy, ph)
    out_w = get_conv_outsize(w, kw, sx, pw)
    dy, dx = 1, 1
    col = cupy.empty((n, c, kh, kw, out_h, out_w), dtype=img.dtype)

    cupy.ElementwiseKernel(
        'raw T img, int32 h, int32 w, int32 out_h, int32 out_w,'
        'int32 kh, int32 kw, int32 sy, int32 sx, int32 ph, int32 pw,'
        'int32 dy, int32 dx',
        'T col',
        '''
           int c0 = i / (kh * kw * out_h * out_w);
           int ky = i / (kw * out_h * out_w) % kh;
           int kx = i / (out_h * out_w) % kw;
           int out_y = i / out_w % out_h;
           int out_x = i % out_w;
           int in_y = ky * dy + out_y * sy - ph;
           int in_x = kx * dx + out_x * sx - pw;
           if (in_y >= 0 && in_y < h && in_x >= 0 && in_x < w) {
             col = img[in_x + w * (in_y + h * c0)];
           } else {
             col = 0;
           }
        ''',
        'im2col')(img.reduced_view(),
                  h, w, out_h, out_w, kh, kw, sy, sx, ph, pw, dy, dx, col)

    return col


def _col2im_gpu(col, sy, sx, ph, pw, h, w):
    """col2im function for GPU.
    This code is ported from Chainer:
    https://github.com/chainer/chainer/blob/v6.4.0/chainer/utils/conv.py
    """
    import cupy

    n, c, kh, kw, out_h, out_w = col.shape
    dx, dy = 1, 1
    img = cupy.empty((n, c, h, w), dtype=col.dtype)

    cupy.ElementwiseKernel(
        'raw T col, int32 h, int32 w, int32 out_h, int32 out_w,'
        'int32 kh, int32 kw, int32 sy, int32 sx, int32 ph, int32 pw,'
        'int32 dx, int32 dy',
        'T img',
        '''
           int c0 = i / (h * w);
           int y  = i / w % h;
           int x  = i % w;
           T val = 0;
           for (int ky = 0; ky < kh; ++ky) {
             int out_y = (y + ph - ky * dy);
             if (0 > out_y || out_y >= out_h * sy) continue;
             if (out_y % sy != 0) continue;
             out_y /= sy;
             for (int kx = 0; kx < kw; ++kx) {
               int out_x = (x + pw - kx * dx);
               if (0 > out_x || out_x >= out_w * sx) continue;
               if (out_x % sx != 0) continue;
               out_x /= sx;
               int k = out_y + out_h * (kx + kw * (ky + kh * c0));
               val = val + col[out_x + out_w * k];
             }
           }
           img = val;
        ''',
        'col2im')(col.reduced_view(),
                  h, w, out_h, out_w, kh, kw, sy, sx, ph, pw, dx, dy, img)
    return img
