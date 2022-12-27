import mnist
import numpy as np

import kraft
import kraft.device
import kraft.optim
import kraft.autograd.ops

from kraft import nn
from kraft.nn import functional as fun

from tqdm import tqdm


class MnistConv(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4),
            nn.ReLU(),
        )

        self.fc = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
            nn.Softmax(),
        )

    def forward(self, xs):
        xs = self.conv(xs)
        xs = self.fc(xs.flatten())

        return xs


def train_epoch(net, device, optimizer, dataset):
    for sample, label in tqdm(dataset):
        inputs = kraft.Variable(np.array(sample, dtype=np.float32) / 255, device=device)
        inputs = inputs.reshape(1, 1, 28, 28)

        target = np.zeros(10)
        target[label] = 1
        target = kraft.Variable(target, device=device)

        outputs = net(inputs)
        loss = fun.ce_loss(outputs, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def test_epoch(net, device, dataset):
    correct_answers = 0

    for sample, label in tqdm(dataset):
        inputs = kraft.Variable(np.array(sample, dtype=np.float32) / 255, device=device)
        inputs = inputs.reshape(1, 1, 28, 28)

        prediction = net(inputs).argmax() + 1

        correct_answers += prediction == label

    print("precision of trained model is", correct_answers / dataset)


def main():
    device = kraft.device.get_gpu_device()

    train = mnist.train_images()
    train_labels = mnist.train_labels()
    test = mnist.test_images()
    test_labels = mnist.test_labels()

    train = list(zip(train, train_labels))
    test = list(zip(test, test_labels))

    net = MnistConv()
    net.to_(device)

    optimizer = kraft.optim.Adam(net.parameters(), lr=5e-3)

    for epoch in range(1, 5):
        train_epoch(net, device, optimizer, train)

    test_epoch(net, device, test)


if __name__ == "__main__":
    main()
