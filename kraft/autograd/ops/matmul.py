import kraft
from kraft.autograd import Function


class MatMul(Function):
    @staticmethod
    def forward(ctx, left, right):
        np = kraft.get_backend(right)

        result = kraft.Variable(np.matmul(left.data, right.data), device=right.device)
        return result

    @staticmethod
    def backward(ctx, grad):
        np = kraft.get_backend(grad)

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


def matmul(left, right):
    return MatMul()(left, right)
