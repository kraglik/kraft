from kraft.nn import Module, Parameter
from kraft.autograd.ops import avg_pool2d


class AvgPool2d(Module):
    def __init__(self, kernel_size, stride=1, pad=0):
        super().__init__()

        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad

    def forward(self, xs):
        return avg_pool2d(
            xs,
            self.kernel_size,
            self.stride,
            self.pad
        )
