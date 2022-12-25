from kraft.nn import Module


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = list(modules)

        for i, module in enumerate(self.modules):
            self._add_entity(str(i), module)

    def forward(self, xs):
        for module in self.modules:
            xs = module(xs)

        return xs
