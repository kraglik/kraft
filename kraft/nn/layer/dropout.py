import kraft

from kraft.nn import Module, Parameter


class Dropout(Module):
    def __init__(self, p: float = 0.5) -> None:
        super().__init__()

        self._p = p
        self._train = True

    def forward(self, inputs: kraft.Variable) -> kraft.Variable:
        if not self.training:
            return inputs

        rand = kraft.randn(list(inputs.shape), dtype=kraft.float32, device=inputs.device)

        return inputs * (rand >= self._p)
