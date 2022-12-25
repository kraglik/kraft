from kraft import Variable


def mse_loss(output: Variable, target: Variable):
    assert output.data.size == target.data.size, "Output and target sizes must be equal"

    error = (output - target)
    error = (error * error).sum()
    error = error * Variable([1.0 / output.data.size], device=error.device)

    return error
