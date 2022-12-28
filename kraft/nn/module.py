from abc import ABC, abstractmethod
from typing import Iterator, Any

from kraft.autograd import Variable
from kraft.device import Device


class Parameter(Variable):
    pass


class Module(object):
    _parameters: dict[str, Parameter]
    _modules: dict[str, 'Module']

    def __init__(self) -> None:
        super().__setattr__("_modules", {})
        super().__setattr__("_parameters", {})
        self._training = True

    def __setattr__(self, key, value):
        params = self.__dict__.get("_parameters")
        modules = self.__dict__.get("_modules")

        if params is None:
            raise AttributeError("super().__init__() method should be called before assignments!")

        if isinstance(value, Parameter):
            params[key] = value

        elif isinstance(value, Module):
            self._modules[key] = value

        return super().__setattr__(key, value)

    @property
    def training(self):
        return self._training

    def train(self):
        self._training = True

        for module in self._modules.values():
            module.train()

    def eval(self):
        self._training = False

        for module in self._modules.values():
            module.eval()

    def parameters(self, recursive: bool = True) -> list[Parameter]:
        parameters = list(self._parameters.values())

        if recursive:
            for module in self._modules.values():
                parameters += module.parameters(recursive=recursive)

        return parameters

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._call_impl(*args, **kwargs)

    def _call_impl(self, *inputs: Any, **kwargs: Any) -> Any:
        return self.forward(*inputs, **kwargs)

    def to_(self, device: Device):
        for parameter in self._parameters.values():
            parameter.to_(device)

        for module in self._modules.values():
            module.to_(device)

    def _add_parameter(self, name, value):
        self._parameters[name] = value

    def _add_module(self, name, value):
        self._modules[name] = value

    def _add_entity(self, name, value):
        if isinstance(value, Parameter):
            self._add_parameter(name, value)

        elif isinstance(value, Module):
            self._add_module(name, value)

    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

