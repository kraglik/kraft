import kraft


def batch_wrapper(loss_function):
    def wrapper(*args, **kwargs):
        batch_loss = loss_function(*args, **kwargs).sum()

        np = kraft.get_backend(batch_loss)

        batch_loss.data /= np.size(args[0].data)

        return batch_loss
    return wrapper
