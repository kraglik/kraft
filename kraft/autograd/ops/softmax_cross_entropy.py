import kraft
from kraft.autograd import Function
from kraft.autograd.utils import logsumexp

from .softmax import softmax


class SoftmaxCrossEntropy(Function):
    @staticmethod
    def forward(ctx, outputs, targets):
        np = kraft.get_backend(outputs)

        batch_n = outputs.shape[0]
        log_z = logsumexp(outputs.data, axis=-1)
        log_p = outputs.data - log_z

        log_p = log_p[np.arange(batch_n), targets.data.ravel()]

        y = (-log_p / batch_n).sum()

        return kraft.Variable(
            data=y,
            device=outputs.device,
            dtype=y.dtype,
            requires_grad=outputs.requires_grad,
        )

    @staticmethod
    def backward(self, grad):
        x, t = self.inputs
        np = kraft.get_backend(t)

        batch_n, class_num = x.shape

        grad *= 1 / batch_n

        y = softmax(x).data

        t_onehot = np.eye(class_num, dtype=t.dtype)[t.data]
        y = (y - t_onehot) * grad

        return y


def softmax_cross_entropy(inputs, targets):
    return SoftmaxCrossEntropy()(inputs, targets)
