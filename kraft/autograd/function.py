from abc import ABC, abstractmethod
from typing import Any

from .utils import broadcast
from .variable import Variable


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

    def drop(self):
        functions = [self]

        while functions:
            fun = functions.pop()
            fun.variable.grad_fn = None
            fun.variable = None

            for var in fun.ctx.inputs:
                if isinstance(var, Variable) and var.grad_fn is not None:
                    functions.append(var.grad_fn)

            fun.ctx.inputs = ()
            fun.ctx.saved_tensors = ()
            fun.ctx.backward_fn = None

    def backward(self):
        grads = self.backward_fn(self.ctx, self.variable.grad)

        if not isinstance(grads, tuple):
            grads = (grads, )

        for inp, grad in zip(self.ctx.input_vars, grads):
            if inp.requires_grad:
                update = broadcast(inp.data, grad)

                if inp.grad is None:
                    inp.grad = update
                else:
                    inp.grad += broadcast(inp.data, grad)


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
