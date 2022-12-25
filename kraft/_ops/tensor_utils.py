import numpy as np

from typing import Type, Optional
from numpy import single

import kraft
from kraft.autograd import Variable
from kraft.device import Device


__all__ = [
    "randn",
    "zeros",
    "zeros_like",
]


def _operation(
    *args,
    function_path: list[str] = [],
    dtype: Type[single] = np.float32,
    requires_grad: bool = True,
    device: Optional[Device] = None,
    **kwargs,
) -> Variable:
    if device is None or device.is_cpu:
        fun = np
        for part in function_path:
            fun = getattr(fun, part)

        data = fun(*args, **kwargs).astype(dtype)
    else:
        if not kraft.device.is_gpu_available():
            raise RuntimeError("GPU is not available!")

        import cupy

        with cupy.cuda.Device(device.index):
            fun = cupy

            for part in function_path:
                fun = getattr(fun, part)

            data = fun(*args, **kwargs).astype(dtype)

    return Variable(data, device=device, dtype=dtype, requires_grad=requires_grad)


def randn(
        shape: int | list[int],
        dtype: Type[single] = np.float32,
        requires_grad: bool = True,
        device: Optional[Device] = None,
) -> Variable:
    if not isinstance(shape, list):
        shape = (shape, )

    return _operation(
        *shape,
        function_path=["random", "randn"],
        dtype=dtype,
        device=device,
        requires_grad=requires_grad,
    )


def zeros(
        shape: int | list[int],
        dtype: Type[single] = np.float32,
        requires_grad: bool = True,
        device: Optional[Device] = None
) -> Variable:
    return _operation(
        shape,
        function_path=["zeros"],
        dtype=dtype,
        device=device,
        requires_grad=requires_grad,
    )


def zeros_like(var: Variable, requires_grad: bool = True) -> Variable:
    return _operation(
        var.data,
        function_path=["zeros_like"],
        dtype=var.dtype,
        device=var.device,
        requires_grad=requires_grad,
    )

