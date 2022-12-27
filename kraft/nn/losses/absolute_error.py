from kraft import Variable


def mae_loss(output: Variable, target: Variable, reduction="mean"):
    assert output.data.size == target.data.size, "Output and target sizes must be equal"
    assert reduction in ("mean", "sum", None), "Unknown reduction type. Expected types are: 'mean', 'sum', None"

    error = (output - target).abs()

    if reduction == "mean":
        error = error.mean()

    elif reduction == "sum":
        error = error.sum()

    return error
