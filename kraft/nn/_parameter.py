from kraft.autograd import Tensor


class Parameter:
    def __init__(
        self,
        data: Tensor,
        requires_grad: bool = True,
    ) -> None:
        self.data = data
        self.requires_grad = requires_grad
