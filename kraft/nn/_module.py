from abc import ABC, abstractmethod
from typing import Iterator, Any

from kraft.autograd import Tensor


class Module(object, ABC):
    _parameters: dict[str, Tensor]
    _modules: dict[str, 'Module']

    def __init__(self) -> None:
        super().__setattr__("_modules", {})
        super().__setattr__("_parameters", {})

    def __setattr__(self, key, value):
        params = self.__dict__.get("_parameters")
        modules = self.__dict__.get("_modules")

        if params is None:
            raise AttributeError("super().__init__() method should be called before assignments!")

        if isinstance(value, Tensor) and value.requires_grad:
            params[key] = value

        elif isinstance(value, Module):
            self._modules[key] = value

        return super(self).__setattr__(key, value)

    def parameters(self, recursive: bool = True) -> Iterator[Tensor]:
        yield from self.__parameters

        if recursive:
            for module in self.__modules:
                yield from module.parameters(recursive=recursive)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._call_impl(*args, **kwargs)

    def _call_impl(self, *inputs: Any, **kwargs: Any) -> Any:
        return self.forward(*inputs, **kwargs)

    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

