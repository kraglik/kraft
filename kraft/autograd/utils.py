from kraft.device.utils import get_backend


def match_shape(x, shape, axis, keepdims):
    np = get_backend(x)

    if shape == ():
        return x, 1

    axis = list(axis) if isinstance(axis, tuple) else axis
    new_shape = np.array(shape)
    new_shape[axis] = 1
    num_reps = np.prod(np.array(shape)[axis])
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
