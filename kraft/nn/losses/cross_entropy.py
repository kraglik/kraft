from kraft import Variable


def ce_loss(output: Variable, target: Variable):
    assert output.data.size == target.data.size, "Output and target sizes must be equal"

    output = output.clip(1e-7, 1 - 1e-7)
    error = output * (-target.log())

    return error
