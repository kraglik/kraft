# Kraft

Kraft is a pythonic deep learning framework with a built-in GPU acceleration.
It is following an early PyTorch, slightly altering its API where necessary.

`kraft.autograd` is an automatic differentiation framework built upon NumPy and CuPy.
It defines `Variable`, `Function`, and some basic functions in the `kraft.autograd.ops` module.

`kraft.optim` contains simple, easy to follow implementations of several popular optimization algorithms, such as Adam and SGD.

`kraft.nn` provides a `Module` class.
Every neural network built with `kraft` must inherit this class.
`kraft.nn` also provides some basic layers, such as `Linear`, `Conv2d`, `MaxPool2d`, and `AvgPool2d`.


## Automatic Differentiation

Kraft uses chain rule under the hood. There's a simplified implementation of this rule in the `_overview/one_file_framework.py` file, Kraft mostly follows ideas expressed in that file.
