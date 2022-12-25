import numpy as np

import kraft


def im2col_array(img, kernel_size, stride, pad, to_matrix=True):
    N, C, H, W = img.shape
    KH, KW = pair(kernel_size)
    SH, SW = pair(stride)
    PH, PW = pair(pad)

    OH = get_conv_outsize(H, KH, SH, PH)
    OW = get_conv_outsize(W, KW, SW, PW)

    if isinstance(img, np.ndarray):
        col = im2col(img, KH, KW, stride=stride, pad=pad)
    else:
        col = im2col_gpu(img, kernel_size, stride, pad)

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

    if isinstance(col, np.ndarray):
        return col2im(col, img_shape, hf=KH, wf=KW, stride=stride, pad=pad)
    else:
        return col2im_gpu(col, SH, SW, PH, PW, H, W)


def get_indices(np, x_shape, hf, wf, stride, pad):
    m, n_c, n_h, n_w = x_shape

    out_h = int((n_h + 2 * pad - hf) / stride) + 1
    out_w = int((n_w + 2 * pad - wf) / stride) + 1

    level1 = np.repeat(np.arange(hf), wf)
    level1 = np.tile(level1, n_c)
    every_levels = stride * np.repeat(np.arange(out_h), out_w)
    i = level1.reshape(-1, 1) + every_levels.reshape(1, -1)

    slide1 = np.tile(np.arange(wf), hf)
    slide1 = np.tile(slide1, n_c)
    every_slides = stride * np.tile(np.arange(out_w), out_h)
    j = slide1.reshape(-1, 1) + every_slides.reshape(1, -1)
    d = np.repeat(np.arange(n_c), hf * wf).reshape(-1, 1)

    return i, j, d


def im2col(x, hf, wf, stride, pad):
    np = kraft.get_backend(x)

    x_padded = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')
    i, j, d = get_indices(np, x.shape, hf, wf, stride, pad)
    cols = x_padded[:, d, i, j]
    cols = np.concatenate(cols, axis=-1)

    return cols


def col2im(dx_col, x_shape, hf, wf, stride, pad):
    np = kraft.get_backend(dx_col)

    n, d, h, w = x_shape

    h_padded, w_padded = h + 2 * pad, w + 2 * pad
    x_padded = np.zeros((n, d, h_padded, w_padded))

    i, j, d = get_indices(np, x_shape, hf, wf, stride, pad)
    d_x_col_reshaped = np.array(np.hsplit(dx_col, n))
    np.add.at(x_padded, (slice(None), d, i, j), d_x_col_reshaped)

    if pad == 0:
        return x_padded
    elif type(pad) is int:
        return x_padded[pad:-pad, pad:-pad, :, :]


def im2col_gpu(img, kernel_size, stride, pad):
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


def col2im_gpu(col, sy, sx, ph, pw, h, w):
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


def get_deconv_outsize(size, k, s, p):
    return s * (size - 1) + k - 2 * p


def get_conv_outsize(input_size, kernel_size, stride, pad):
    return (input_size + pad * 2 - kernel_size) // stride + 1


def pair(x):
    if isinstance(x, int):
        return (x, x)
    elif isinstance(x, tuple):
        assert len(x) == 2
        return x
    else:
        raise ValueError
