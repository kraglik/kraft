from kraft import Variable


def ce_loss(output: Variable, target: Variable):
    assert output.data.size == target.data.size, "Output and target sizes must be equal"
    assert 0 <= output.data <= 1, "Expected data to be in between 0 and 1"

    output = output.clip(1e-7, 1 - 1e-7)
    error = output * (-target.log())

    return error
