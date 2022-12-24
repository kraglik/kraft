from typing import Iterator

from ._parameter import Parameter


class Module(object):
    def __init__(self) -> None:
        self.__parameters: list[Parameter] = []
        self.__modules: list[Module] = []
        self.__init_called = True

    def __setattr__(self, key, value):
        if not getattr(self, '__init_called', False):
            raise RuntimeError("__init__() method should be called in a Module!")

        if isinstance(value, Parameter):
            self.__parameters.append(value)

        elif isinstance(value, Module):
            self.__modules.append(value)

        return super(self).__setattr__(key, value)

    def parameters(self, recursive: bool = True) -> Iterator[Parameter]:
        yield from self.__parameters

        if recursive:
            for module in self.__modules:
                yield from module.parameters(recursive=recursive)
