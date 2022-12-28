import kraft
from kraft import Variable
from kraft.autograd.ops import softmax_cross_entropy


def ce_loss(output: Variable, target: Variable, reduction="mean"):
    np = kraft.get_backend(output)

    batch_n = output.shape[0]
    log_z = _logsumexp(output, axis=-1)
    log_p = output - log_z

    result = -log_p[np.arange(batch_n), target.data.ravel()]

    if reduction == "mean":
        return result.mean()

    elif reduction == "sum":
        return result.sum()

    return result

    # return softmax_cross_entropy(output, target)


def _logsumexp(x, axis=1):
    m = x.max(axis=axis, keep_dims=True)
    y = x - m
    y = y.exp()
    s = y.sum(axis=axis, keep_dims=True)
    s = s.log()
    m = m + s

    return m
