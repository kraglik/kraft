from kraft import Variable
from kraft.autograd.ops import softmax_cross_entropy


def ce_loss(output: Variable, target: Variable):
    assert output.data.size == target.data.size, "Output and target sizes must be equal"

    output = output.clip(1e-9, 1 - 1e-9)
    batch_size = output.shape[0]

    return -(target * (output + 1e-7).log()).sum() / batch_size

    # return softmax_cross_entropy(output, target)
