import numpy as np

import kraft
from kraft.autograd.ops import conv2d
from kraft.autograd.ops.utils.conv import pair
from kraft.nn import Module, Parameter


class Conv2d(Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        pad=0,
        nobias=False,
        dtype=kraft.float32,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad
        self.dtype = dtype

        self.W = self._init_weights()

        if nobias:
            self.b = None
        else:
            self.b = Parameter(np.zeros(out_channels, dtype=dtype))

    def _init_weights(self):
        C, OC = self.in_channels, self.out_channels
        KH, KW = pair(self.kernel_size)
        scale = np.sqrt(1 / (C * KH * KW))
        weights_data = kraft.randn([OC, C, KH, KW], dtype=self.dtype) * scale

        return weights_data

    def forward(self, x):
        y = conv2d(x, self.W, self.b, self.stride, self.pad)
        return y
