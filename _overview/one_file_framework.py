import random
from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class Variable(object):
    def __init__(self, data: Any):
        self.data = self._to_np_ndarray(data)
        self.grad = None
        self.grad_fn = None

        self.zero_grad()

    @property
    def shape(self):
        return self.data.shape

    @staticmethod
    def _to_np_ndarray(data):
        if isinstance(data, Variable):
            data = data.data

        return np.array(data, dtype=np.float32) if not isinstance(data, np.ndarray) else data

    @staticmethod
    def _to_variable(var):
        return var if isinstance(var, Variable) else Variable(var)

    def exp(self):
        return exp(self)

    def __neg__(self):
        return neg(self)

    def __truediv__(self, other):
        return div(self, self._to_variable(other))

    def __rtruediv__(self, other):
        return div(Variable._to_variable(other), self)

    def __add__(self, other):
        return add(self, self._to_variable(other))

    def __radd__(self, other):
        return add(Variable._to_variable(other), self)

    def __sub__(self, other):
        return sub(self, self._to_variable(other))

    def __rsub__(self, other):
        return sub(self._to_variable(other), self)

    def __mul__(self, other):
        return mul(self, self._to_variable(other))

    def __rmul__(self, other):
        return mul(self._to_variable(other), self)

    def __matmul__(self, other):
        return matmul(self, self._to_variable(other))

    def __hash__(self):
        return id(self)

    def __rmatmul__(self, other):
        return matmul(self._to_variable(other), self)

    def __eq__(self, other):
        return Variable((self.data == self._to_variable(other).data).astype(np.float32))

    def __ne__(self, other):
        return Variable((self.data != self._to_variable(other).data).astype(np.float32))

    def __lt__(self, other):
        return Variable((self.data < self._to_variable(other).data).astype(np.float32))

    def __gt__(self, other):
        return Variable((self.data > self._to_variable(other).data).astype(np.float32))

    def __le__(self, other):
        return Variable((self.data <= self._to_variable(other).data).astype(np.float32))

    def __ge__(self, other):
        return Variable((self.data >= self._to_variable(other).data).astype(np.float32))

    def sum(self):
        return sum_var(self)

    def zero_grad(self):
        self.grad = np.zeros_like(self.data)

    def backward(self, grad=None):
        grad = self._to_np_ndarray(grad) if grad else np.ones_like(self.data)
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
        return f"<Variable ({data=}), {grad_fn=}>"


class FunctionCtx(object):
    def __init__(self, inputs):
        self.inputs = inputs
        self.input_vars = tuple(i for i in inputs if isinstance(i, Variable))
        self.saved_tensors = ()

    def save_for_backward(self, *tensors):
        self.saved_tensors = tensors


class BackwardFunction(object):
    def __init__(self, ctx, backward, variable):
        self.ctx = ctx
        self.backward_fn = backward
        self.variable = variable

    def backward(self):
        grads = self.backward_fn(self.ctx, self.variable.grad)

        if not isinstance(grads, tuple):
            grads = (grads, )

        for inp, grad in zip(self.ctx.input_vars, grads):
            inp.grad += broadcast(inp.grad, grad)


class Function(ABC):
    def __call__(self, *inputs):
        inputs = tuple(inputs)
        ctx = FunctionCtx(inputs=inputs)
        output = self.forward(ctx, *inputs)
        output.grad_fn = BackwardFunction(
            ctx=ctx,
            backward=self.backward,
            variable=output,
        )
        return output

    @staticmethod
    @abstractmethod
    def forward(ctx, *inputs):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def backward(ctx, grad):
        raise NotImplementedError


class Exp(Function):
    @staticmethod
    def forward(ctx, var):
        output = Variable(np.exp(var.data))
        ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx, grad):
        output, = ctx.saved_tensors
        return output.data * grad


class Div(Function):
    @staticmethod
    def forward(ctx, left, right):
        output = Variable(left.data / right.data)
        ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx, grad):
        left, right = ctx.inputs
        output, = ctx.saved_tensors

        left_grad = grad.data / right.data
        right_grad = -grad * output.data ** 2

        return left_grad, right_grad


class Neg(Function):
    @staticmethod
    def forward(ctx, var):
        return Variable(-var.data)

    @staticmethod
    def backward(ctx, grad):
        return -grad


class Sub(Function):
    @staticmethod
    def forward(ctx, left, right):
        return Variable(left.data - right.data)

    @staticmethod
    def backward(ctx, grad):
        return grad, -grad


class Add(Function):
    @staticmethod
    def forward(ctx, left, right):
        result = Variable(left.data + right.data)
        return result

    @staticmethod
    def backward(ctx, grad):
        return grad, grad


class Mul(Function):
    @staticmethod
    def forward(ctx, left, right):
        return Variable(left.data * right.data)

    @staticmethod
    def backward(ctx, grad):
        left, right = ctx.inputs

        return right.data * grad, left.data * grad


class MatMul(Function):
    @staticmethod
    def forward(ctx, left, right):
        result = Variable(np.matmul(left.data, right.data))
        return result

    @staticmethod
    def backward(ctx, grad):
        left, right = ctx.inputs

        left_t = np.transpose(left.data)
        right_t = np.transpose(right.data)

        left_grad, right_grad = grad, grad

        if len(left_t.shape) == 1 and len(right_t.shape) == 2:
            right_grad = right_grad.reshape((1, right_grad.size))
            left_t = left_t.reshape((left_t.size, 1))

        left_grad = left_grad @ right_t
        right_grad = left_t @ right_grad

        return left_grad, right_grad


class Sum(Function):
    @staticmethod
    def forward(ctx, var, axis, keepdims):
        return Variable(np.sum(var.data))

    @staticmethod
    def backward(ctx, grad):
        var, axis, keepdims = ctx.inputs

        sum_grad = match_shape(grad, var.shape, axis, keepdims)[0]

        return sum_grad


def exp(var):
    return Exp()(var)


def div(left, right):
    return Div()(left, right)


def neg(var):
    return Neg()(var)


def add(left, right):
    return Add()(left, right)


def sub(left, right):
    return Sub()(left, right)


def mul(left, right):
    return Mul()(left, right)


def matmul(left, right):
    return MatMul()(left, right)


def sum_var(var, axis=None, keepdims=False):
    return Sum()(var, axis, keepdims)


def match_shape(x, shape, axis, keepdims):
    if shape == ():
        return x, 1
    axis = list(axis) if isinstance(axis, tuple) else axis
    new_shape = np.array(shape)
    new_shape[axis] = 1
    num_reps = np.prod(np.array(shape)[axis])
    return np.reshape(x, new_shape) + np.zeros(shape, dtype=np.float32), num_reps


def broadcast(target_grad, input_grad):
    while np.ndim(input_grad) > np.ndim(target_grad):
        input_grad = np.sum(input_grad, axis=0)
    for axis, dim in enumerate(np.shape(target_grad)):
        if dim == 1:
            input_grad = np.sum(input_grad, axis=axis, keepdims=True)
    return input_grad


class Parameter(Variable):
    pass


class Module(object):
    _parameters: dict[str, Parameter]
    _modules: dict[str, 'Module']

    def __init__(self) -> None:
        super().__setattr__("_modules", {})
        super().__setattr__("_parameters", {})

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

    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError


class Linear(Module):
    def __init__(self, input_dims, output_dims):
        super().__init__()

        self.weights = Parameter(np.random.rand(input_dims, output_dims) * 0.1 - 0.05)
        self.bias = Parameter(np.random.rand(output_dims) * 0.1 - 0.05)

    def forward(self, xs: Variable) -> Variable:
        return xs @ self.weights + self.bias


class MLP(Module):
    def __init__(self):
        super().__init__()

        self.l1 = Linear(2, 32)
        self.l2 = Linear(32, 32)
        self.l3 = Linear(32, 1)

    def forward(self, xs):
        xs = relu(self.l1(xs))
        xs = relu(self.l2(xs))
        xs = sigmoid(self.l3(xs))

        return xs


class SGD(object):
    def __init__(self, parameters, lr=1e-2, nesterov=False, momentum=0.0):
        self.parameters = parameters
        self._lr = lr
        self.momentum = momentum
        self.nesterov = nesterov
        self._cache = {
            parameter: np.zeros(parameter.data.shape)
            for parameter in self.parameters
        }

    def zero_grad(self):
        for param in self.parameters:
            param.zero_grad()

    def step(self):
        if self.nesterov:
            for parameter in self.parameters:
                parameter.data -= self._cache[parameter] * self.momentum * self._lr

            for parameter in self.parameters:
                dw = parameter.grad
                self._cache[parameter] *= self.momentum
                self._cache[parameter] += dw * (1.0 - self.momentum)
                parameter.data -= dw * (1.0 - self.momentum) * self._lr

        else:
            for parameter in self.parameters:
                dw = parameter.grad
                self._cache[parameter] *= self.momentum
                self._cache[parameter] += dw * (1.0 - self.momentum)
                parameter.data -= self._cache[parameter] * self._lr


def mse_loss(output: Variable, target: Variable):
    assert output.data.size == target.data.size, "Output and target sizes must be equal"

    error = (output - target)
    error = (error * error).sum()
    error = error * Variable([1.0 / output.data.size])

    return error


def sigmoid(var):
    return 1.0 / (1.0 + (-var).exp())


def relu(t: Variable):
    return t * (t >= 0.0) + t * (t >= 0.0)


def main():
    inputs = Variable([1.0, 0.0])
    network = MLP()
    optimizer = SGD(network.parameters(), lr=5e-1, nesterov=True, momentum=0.15)

    data = [
        ([1.0, 1.0], 0.0),
        ([1.0, 0.0], 1.0),
        ([0.0, 1.0], 1.0),
        ([0.0, 0.0], 0.0),
    ]

    for _ in range(1000):
        for _ in range(3):
            inputs, target = random.choice(data)
            optimizer.zero_grad()

            inputs = Variable(inputs)
            outputs = network(inputs)

            loss = mse_loss(outputs, Variable(target))
            loss.backward()

        optimizer.step()

    for inputs, target in data:
        inputs = Variable(inputs)
        outputs = network(inputs)

        print(inputs.data, outputs.data, target)

        # Prints
        # [1. 1.] [0.01827366] 0.0
        # [1. 0.] [0.98852127] 1.0
        # [0. 1.] [0.98853063] 1.0
        # [0. 0.] [0.01063632] 0.0


if __name__ == "__main__":
    main()
