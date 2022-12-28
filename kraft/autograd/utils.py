import kraft
from kraft.device.utils import get_backend


def match_shape(x, shape, axis, keepdims):
    np = get_backend(x)

    if shape == ():
        return x, 1

    axis = list(axis) if isinstance(axis, tuple) else axis
    new_shape = np.array(shape)
    num_reps = np.prod(new_shape[axis])
    new_shape[axis] = 1
    new_shape = tuple(new_shape.tolist())
    shape = tuple(shape)
    return np.reshape(x, new_shape) + np.zeros(shape, dtype=np.float32), num_reps


def broadcast(target_grad, input_grad):
    np = get_backend(target_grad)

    while np.ndim(input_grad) > np.ndim(target_grad):
        input_grad = np.sum(input_grad, axis=0)
    for axis, dim in enumerate(np.shape(target_grad)):
        if dim == 1:
            input_grad = np.sum(input_grad, axis=axis, keepdims=True)
    return input_grad


def broadcast_to(target_shape, input_grad):
    np = get_backend(input_grad)

    while np.ndim(input_grad) > len(target_shape):
        input_grad = np.sum(input_grad, axis=0)
    for axis, dim in enumerate(target_shape):
        if dim == 1:
            input_grad = np.sum(input_grad, axis=axis, keepdims=True)
    return input_grad


def logsumexp(x, axis=1):
    xp = kraft.get_backend(x)
    m = x.max(axis=axis, keepdims=True)
    y = x - m
    xp.exp(y, out=y)
    s = y.sum(axis=axis, keepdims=True)
    xp.log(s, out=s)
    m += s

    return m

