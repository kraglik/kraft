from typing import Any, Optional, Type

import numpy as np
from numpy import single

import kraft.device.utils
from kraft.device import Device


class Variable(object):
    def __init__(
            self,
            data: Any,
            dtype=None,
            device: Optional[Device] = None,
            requires_grad=True,
    ):
        self.device = device
        self.grad = None
        self.grad_fn = None
        self.data = self._to_ndarray(data, dtype)
        self.dtype = dtype
        self.requires_grad = requires_grad

        self.zero_grad()

    @property
    def shape(self):
        return self.data.shape

    def _to_ndarray(self, data, dtype):
        if self.device is None or self.device.is_cpu:
            return self._to_np_ndarray(data, dtype)

        return self._to_cp_ndarray(data, dtype)

    @staticmethod
    def _to_np_ndarray(data, dtype):
        if isinstance(data, Variable):
            data = data.data

        if kraft.device.is_gpu_available():
            import cupy

            if isinstance(data, cupy.ndarray):
                data = data.get()

        return np.array(data, dtype=dtype) if not isinstance(data, np.ndarray) else data

    def _to_cp_ndarray(self, data, dtype):
        if not kraft.device.is_gpu_available():
            raise RuntimeError("GPU is not available!")

        import cupy

        with cupy.cuda.Device(self.device.index):
            return cupy.array(data, dtype=dtype)

    @staticmethod
    def _to_variable(var, device=None):
        return var if isinstance(var, Variable) else Variable(var, device=device)

    def to_(self, device: Device):
        self.device = device
        self.data = self._to_ndarray(self.data, self.dtype)

        if self.grad is not None:
            self.grad = self._to_ndarray(self.grad, self.grad.dtype)

    def exp(self):
        from kraft.autograd.ops import exp

        return exp(self)

    def __neg__(self):
        from kraft.autograd.ops import neg

        return neg(self)

    def __truediv__(self, other):
        from kraft.autograd.ops import div

        return div(self, self._to_variable(other, device=self.device))

    def __rtruediv__(self, other):
        from kraft.autograd.ops import div

        return div(Variable._to_variable(other, device=self.device), self)

    def __add__(self, other):
        from kraft.autograd.ops import add

        return add(self, self._to_variable(other, device=self.device))

    def __radd__(self, other):
        from kraft.autograd.ops import add

        return add(Variable._to_variable(other, device=self.device), self)

    def __sub__(self, other):
        from kraft.autograd.ops import sub

        return sub(self, self._to_variable(other, device=self.device))

    def __rsub__(self, other):
        from kraft.autograd.ops import sub

        return sub(self._to_variable(other, device=self.device), self)

    def __mul__(self, other):
        from kraft.autograd.ops import mul

        return mul(self, self._to_variable(other, device=self.device))

    def __rmul__(self, other):
        from kraft.autograd.ops import mul

        return mul(self._to_variable(other, device=self.device), self)

    def __matmul__(self, other):
        from kraft.autograd.ops import matmul

        return matmul(self, self._to_variable(other, device=self.device))

    def __rmatmul__(self, other):
        from kraft.autograd.ops import matmul

        return matmul(self._to_variable(other, device=self.device), self)

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return Variable(
            (self.data == self._to_ndarray(other, dtype=self.dtype)).astype(np.float32),
            device=self.device,
        )

    def __ne__(self, other):
        return Variable(
            (self.data != self._to_ndarray(other, dtype=self.dtype)).astype(np.float32),
            device=self.device,
        )

    def __lt__(self, other):
        return Variable(
            (self.data < self._to_ndarray(other, dtype=self.dtype)).astype(np.float32),
            device=self.device,
        )

    def __gt__(self, other):
        return Variable(
            (self.data > self._to_ndarray(other, dtype=self.dtype)).astype(np.float32),
            device=self.device,
        )

    def __le__(self, other):
        return Variable(
            (self.data <= self._to_ndarray(other, dtype=self.dtype)).astype(np.float32),
            device=self.device,
        )

    def __ge__(self, other):
        return Variable(
            (self.data >= self._to_ndarray(other, dtype=self.dtype)).astype(np.float32),
            device=self.device,
        )

    def sum(self):
        from kraft.autograd.ops import sum_var

        return sum_var(self)

    def zero_grad(self):
        if self.requires_grad:
            self.grad = np.zeros_like(self.data)

    def backward(self, grad=None):
        grad = (
            self._to_ndarray(grad, self.dtype)
            if grad
            else kraft.get_backend(self.data).ones_like(self.data)
        )
        self.grad += grad

        visited_functions = set()
        functions = [self.grad_fn]

        while functions:
            node = functions.pop()

            if node is None:
                continue

            node.backward()

            for var in node.ctx.input_vars:
                if var.grad_fn not in visited_functions:
                    functions.append(var.grad_fn)
                    visited_functions.add(var.grad_fn)

    def __str__(self):
        grad_fn = self.grad_fn
        data = self.data
        device = self.device

        return f"<kraft.Variable ({data=}), {device=}, {grad_fn=}>"
