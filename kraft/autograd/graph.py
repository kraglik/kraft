from typing import Any

from .function import Function


class _ComputationalGraphNode:
    def __init__(self, inputs: Any, outputs: Any, function: 'Function') -> None:
        self._input = inputs
        self._output = outputs
        self._function = function

    @property
    def inputs(self) -> Any:
        return self._input

    @property
    def outputs(self) -> Any:
        return self._output

    @property
    def function(self) -> 'Function':
        return self._function
