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
    np = kraft.get_backend(input_grad)

    return np.broadcast_to(input_grad, target_grad.shape)

    # return broadcast_to(target_grad.shape, input_grad)


def broadcast_to(target_shape, input_grad):
    np = get_backend(input_grad)

    while np.ndim(input_grad) > len(target_shape):
        input_grad = np.sum(input_grad, axis=0)

    for axis, dim in enumerate(target_shape):
        if dim == 1:
            input_grad = np.sum(input_grad, axis=axis, keepdims=True)

    return np.ascontiguousarray(input_grad)


def logsumexp(x, axis=1):
    np = kraft.get_backend(x)
    m = x.max(axis=axis, keepdims=True)
    y = x - m
    np.exp(y, out=y)
    s = y.sum(axis=axis, keepdims=True)
    np.log(s, out=s)
    m += s

    return np.ascontiguousarray(m)


def array_sum_to(x, shape):
    ndim = len(shape)
    lead = x.ndim - ndim
    lead_axis = tuple(range(lead))

    axis = tuple([i + lead for i, sx in enumerate(shape) if sx == 1])
    y = x.sum(lead_axis + axis, keepdims=True)

    if lead > 0:
        y = y.squeeze(lead_axis)

    return y

