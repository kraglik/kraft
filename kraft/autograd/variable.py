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
        dtype = dtype or getattr(data, "dtype", None)

        if self.device is None or self.device.is_cpu:
            return self._to_np_ndarray(data, dtype)

        return self._to_cp_ndarray(data, dtype)

    @staticmethod
    def _to_np_ndarray(data, dtype):
        if isinstance(data, Variable):
            data = data.data

        if isinstance(data, np.ndarray):
            return data

        if kraft.device.is_gpu_available():
            import cupy

            if isinstance(data, cupy.ndarray):
                data = data.get()

        return np.array(data, dtype=dtype) if not isinstance(data, np.ndarray) else data

    def _to_cp_ndarray(self, data, dtype):
        if not kraft.device.is_gpu_available():
            raise RuntimeError("GPU is not available!")

        cupy = kraft.device.get_gpu_backend()

        if isinstance(data, cupy.ndarray) and data.device == self.device.device_info:
            return data

        with cupy.cuda.Device(self.device.index):
            return cupy.array(data, dtype=dtype)

    @staticmethod
    def _to_variable(var, device=None, requires_grad=True):
        return (
            var
            if isinstance(var, Variable)
            else Variable(
                var,
                device=device,
                requires_grad=requires_grad
            )
        )

    def to_(self, device: Device):
        self.device = device
        self.data = self._to_ndarray(self.data, self.dtype)

        if self.grad is not None:
            self.grad = self._to_ndarray(self.grad, self.grad.dtype)

    def exp(self):
        from kraft.autograd.ops import exp

        return exp(self)

    def __getitem__(self, index):
        from kraft.autograd.ops import slice_

        return slice_(self, index)

    def __neg__(self):
        from kraft.autograd.ops import neg

        return neg(self)

    def __truediv__(self, other):
        from kraft.autograd.ops import div, div_var_float

        if isinstance(other, (int, float)):
            return div_var_float(self, other)

        return div(
            self,
            self._to_variable(
                other,
                device=self.device,
                requires_grad=self.requires_grad or other.requires_grad,
            )
        )

    def __rtruediv__(self, other):
        from kraft.autograd.ops import div, div_float_var

        if isinstance(other, (int, float)):
            return div_float_var(other, self)

        return div(
            Variable._to_variable(
                other,
                device=self.device,
                requires_grad=self.requires_grad or other.requires_grad,
            ),
            self
        )

    def __add__(self, other):
        from kraft.autograd.ops import add, add_var_float

        if isinstance(other, (float, int)):
            return add_var_float(self, other)

        return add(
            self,
            self._to_variable(
                other,
                device=self.device,
                requires_grad=self.requires_grad or other.requires_grad,
            )
        )

    def __radd__(self, other):
        from kraft.autograd.ops import add, add_var_float

        if isinstance(other, (float, int)):
            return add_var_float(self, other)

        return add(
            Variable._to_variable(
                other,
                device=self.device,
                requires_grad=self.requires_grad or other.requires_grad,
            ),
            self
        )

    def __sub__(self, other):
        from kraft.autograd.ops import sub, sub_var_float

        if isinstance(other, (float, int)):
            return sub_var_float(self, other)

        return sub(
            self,
            self._to_variable(
                other,
                device=self.device,
                requires_grad=self.requires_grad or other.requires_grad,
            )
        )

    def __rsub__(self, other):
        from kraft.autograd.ops import sub, sub_float_var

        if isinstance(other, (float, int)):
            return sub_float_var(other, self)

        return sub(
            self._to_variable(
                other,
                device=self.device,
                requires_grad=self.requires_grad or other.requires_grad,
            ),
            self
        )

    def __mul__(self, other):
        from kraft.autograd.ops import mul, mul_var_float

        if isinstance(other, (float, int)):
            return mul_var_float(self, other)

        return mul(
            self,
            self._to_variable(
                other,
                device=self.device,
                requires_grad=self.requires_grad or other.requires_grad,
            )
        )

    def __rmul__(self, other):
        from kraft.autograd.ops import mul, mul_var_float

        if isinstance(other, (float, int)):
            return mul_var_float(self, other)

        return mul(
            self._to_variable(
                other,
                device=self.device,
                requires_grad=self.requires_grad or other.requires_grad,
            ),
            self
        )

    def __matmul__(self, other):
        from kraft.autograd.ops import matmul

        return matmul(
            self,
            self._to_variable(
                other,
                device=self.device,
                requires_grad=self.requires_grad or other.requires_grad,
            )
        )

    def __rmatmul__(self, other):
        from kraft.autograd.ops import matmul

        return matmul(
            self._to_variable(
                other,
                device=self.device,
                requires_grad=self.requires_grad or other.requires_grad,
            ),
            self
        )

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return Variable(
            (self.data == self._to_ndarray(other, dtype=self.dtype)).astype(np.float32),
            device=self.device,
            requires_grad=self.requires_grad or other.requires_grad,
        )

    def __ne__(self, other):
        return Variable(
            (self.data != self._to_ndarray(other, dtype=self.dtype)).astype(np.float32),
            device=self.device,
            requires_grad=self.requires_grad or other.requires_grad,
        )

    def __lt__(self, other):
        return Variable(
            (self.data < self._to_ndarray(other, dtype=self.dtype)).astype(np.float32),
            device=self.device,
            requires_grad=self.requires_grad or other.requires_grad,
        )

    def __gt__(self, other):
        return Variable(
            (self.data > self._to_ndarray(other, dtype=self.dtype)).astype(np.float32),
            device=self.device,
            requires_grad=self.requires_grad or other.requires_grad,
        )

    def __le__(self, other):
        return Variable(
            (self.data <= self._to_ndarray(other, dtype=self.dtype)).astype(np.float32),
            device=self.device,
            requires_grad=self.requires_grad or other.requires_grad,
        )

    def __ge__(self, other):
        if not isinstance(other, (float, int)):
            other = self._to_ndarray(other, dtype=self.dtype)

        return Variable(
            (self.data >= other).astype(np.float32),
            device=self.device,
            requires_grad=self.requires_grad or other.requires_grad,
        )

    def argmax(self, axis=-1):
        return np.argmax(self.data, axis=axis)

    def item(self):
        return self.data.item()

    def sum(self, axis=None, keep_dims=False):
        from kraft.autograd.ops import sum_var

        return sum_var(self, axis=axis, keep_dims=keep_dims)

    def tanh(self):
        from kraft.autograd.ops import tanh

        return tanh(self)

    def reshape(self, *new_shape):
        from kraft.autograd.ops import reshape

        return reshape(self, tuple(new_shape))

    def flatten(self):
        from kraft.autograd.ops import flatten

        return flatten(self)

    def sqrt(self):
        from kraft.autograd.ops import sqrt

        return sqrt(self)

    def square(self):
        from kraft.autograd.ops import square

        return square(self)

    def clip(self, minimum, maximum):
        from kraft.autograd.ops import clip

        return clip(self, minimum, maximum)

    def abs(self):
        from kraft.autograd.ops import abs_

        return abs_(self)

    def log(self):
        from kraft.autograd.ops import log

        return log(self)

    def mean(self, axis=None, keep_dims=True):
        from kraft.autograd.ops import mean

        return mean(self, axis, keep_dims)

    def min(self, axis=None, keep_dims=False):
        from kraft.autograd.ops import min_

        return min_(self, axis, keep_dims)

    def max(self, axis=None, keep_dims=False):
        from kraft.autograd.ops import max_

        return max_(self, axis, keep_dims)

    def zero_grad(self):
        if self.grad_fn is not None:
            self.grad_fn.drop()

        del self.grad
        del self.grad_fn

        self.grad_fn = None
        self.grad = None

    def backward(self, grad=None):
        grad = (
            self._to_ndarray(grad, self.dtype)
            if grad
            else kraft.get_backend(self.data).ones_like(self.data)
        )
        if self.grad is None:
            self.grad = grad
        else:
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
