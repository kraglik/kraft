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

        self.W = Parameter(self._init_weights())

        if nobias:
            self.b = None
        else:
            self.b = Parameter(kraft.randn([1, out_channels, 1, 1], dtype=dtype).data * 0.05 - 0.025)

    def _init_weights(self):
        C, OC = self.in_channels, self.out_channels
        KH, KW = pair(self.kernel_size)
        # scale = np.sqrt(1 / (C * KH * KW))
        weights_data = kraft.randn([OC, C, KH, KW], dtype=self.dtype).data * 0.05 - 0.025

        return weights_data

    def forward(self, x):
        y = conv2d(x, self.W, self.stride, self.pad)
        return y + self.b
