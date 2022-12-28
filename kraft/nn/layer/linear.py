import kraft

from kraft.nn import Module, Parameter


class Linear(Module):
    def __init__(self, input_dims: int, output_dims: int, bias: bool = True) -> None:
        super().__init__()

        self.weights = Parameter(
            kraft.randn(
                [input_dims, output_dims],
                dtype=kraft.float32,
                requires_grad=True,
            ).data * 0.05 - 0.025
        )

        if bias:
            self.bias = Parameter(
                kraft.randn(
                    output_dims,
                    dtype=kraft.float32,
                    requires_grad=True,
                ).data * 0.05 - 0.025
            )
        else:
            self.bias = Parameter(kraft.zeros(output_dims, dtype=kraft.float32).data, requires_grad=False)

    def forward(self, inputs: kraft.Variable) -> kraft.Variable:
        return inputs @ self.weights + self.bias
