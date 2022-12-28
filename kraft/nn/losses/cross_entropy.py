from kraft import Variable
from kraft.autograd.ops import softmax_cross_entropy


def ce_loss(output: Variable, target: Variable):
    # output = output.clip(1e-9, 1 - 1e-9)
    # batch_size = output.shape[0]

    # return -(output * (target + 1e-5).log()).sum()

    return softmax_cross_entropy(output, target)
