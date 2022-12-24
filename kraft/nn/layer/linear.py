import kraft

from kraft.nn import Module


class Linear(Module):
    def __init__(self, input_dims: int, output_dims: int, bias: bool = True) -> None:
        super().__init__()

        self.weights = kraft.randn(
            input_dims,
            output_dims,
            dtype=kraft.float32,
            requires_grad=True,
        ) * 0.05 - 0.025

        if bias:
            self.bias = kraft.randn(
                output_dims,
                dtype=kraft.float32,
                requires_grad=True,
            ) * 0.05 - 0.025
        else:
            self.bias = kraft.zeros(output_dims, dtype=kraft.float32, requires_grad=False)

    def forward(self, inputs: kraft.Tensor) -> kraft.Tensor:
        return inputs @ self.weights + self.bias
