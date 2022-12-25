import random

import kraft
import kraft.device
import kraft.optim

from kraft import nn
from kraft.nn import functional as fun


class MLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, xs):
        return self.network(xs)


DATA = [
    ([0.0, 0.0], 0.0),
    ([0.0, 1.0], 1.0),
    ([1.0, 0.0], 1.0),
    ([1.0, 1.0], 0.0),
]


def main():
    device = kraft.device.get_gpu_device()
    net = MLP()
    net.to_(device)

    sgd = kraft.optim.Adam(net.parameters(), lr=1e-2)

    n_get_backend(parameter)epochs = 1500

    for _ in range(n_epochs):
        sgd.zero_grad()

        items = random.choices(DATA, k=1)
        xs = [item[0] for item in items]
        ys = [item[1] for item in items]

        inputs = kraft.Variable(xs, device=device)
        targets = kraft.Variable(ys, device=device)

        outputs = net(inputs)
        loss = fun.mse_loss(outputs, targets)
        loss.backward()

        sgd.step()

    xs = [item[0] for item in DATA]
    ys = [item[1] for item in DATA]

    for xs, y in zip(xs, ys):
        inputs = kraft.Variable(xs, device=device)
        targets = kraft.Variable(y, device=device)

        print(xs, net(inputs).data, y)


if __name__ == "__main__":
    main()
