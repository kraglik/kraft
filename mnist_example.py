import random
from collections import Counter

import mnist
import numpy as np

import kraft
import kraft.device
import kraft.optim
import kraft.autograd.ops

from kraft import nn
from kraft.nn import functional as fun

from tqdm import tqdm
from more_itertools import chunked


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.residual = nn.Sequential(
            nn.Conv2d(out_channels, out_channels // 2, kernel_size=3, pad=1),
            nn.ReLU(),
            nn.Dropout(p=0.15),
            nn.Conv2d(out_channels // 2, out_channels // 2, kernel_size=3, pad=1),
            nn.ReLU(),
            nn.Dropout(p=0.15),
            nn.Conv2d(out_channels // 2, out_channels, kernel_size=3, pad=1),
        )

    def forward(self, xs):
        xs = xs + self.residual(xs)

        return fun.relu(xs)


class MnistConv(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=7, bias=True),
            ResidualBlock(8, 8),

            nn.MaxPool2d(kernel_size=2, stride=2),

            ResidualBlock(8, 8),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=4, bias=True),
            ResidualBlock(16, 16),

            nn.MaxPool2d(kernel_size=2, stride=2),

            ResidualBlock(16, 16),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, bias=True),
            ResidualBlock(32, 32),

            ResidualBlock(32, 32),
        )

        self.fc = nn.Sequential(
            nn.Linear(32, 64, bias=True),
            nn.ReLU(),
            nn.Linear(64, 10, bias=True),
        )

    def forward(self, xs):
        xs = self.conv(xs)
        xs = self.fc(xs.flatten())

        return xs


def inputs_targets_from_chunk(chunk, device):
    inputs = []
    targets = []

    for i, t in chunk:
        input = np.array(i, dtype=np.float32)
        input = input.reshape((1, 1, 28, 28))
        inputs.append(input / 255)

        target = np.array([[t]])
        targets.append(target)

    inputs = kraft.Variable(
        np.concatenate(inputs, axis=0),
        device=device,
        requires_grad=False,
    )
    target = kraft.Variable(
        np.concatenate(targets, axis=0),
        device=device,
        requires_grad=False,
    )

    return inputs, target


def train_epoch(net, device, optimizer, regularizer, dataset):
    for chunk in chunked(tqdm(dataset), 512):
        optimizer.zero_grad()

        inputs, target = inputs_targets_from_chunk(chunk, device)

        outputs = net(inputs)

        loss = fun.ce_loss(outputs, target, reduction="mean")
        # loss = loss + regularizer.get_loss(net.parameters())

        loss.backward()
        optimizer.step()


def test_epoch(net, device, dataset):
    correct_answers = 0

    answers = Counter()

    for sample, label in tqdm(dataset):
        inputs = kraft.Variable(
            np.array(sample, dtype=np.float32) / 255,
            device=device,
            requires_grad=False,
        )
        inputs = inputs.reshape(1, 1, 28, 28)

        prediction = net(inputs).argmax().item()
        answers[prediction] += 1

        correct_answers += prediction == label

    print(answers)
    print("precision of trained model is", correct_answers / len(dataset))


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
    regularizer = nn.L2Regularizer(alpha=1e-2, reduction="mean")

    # optimizer = kraft.optim.SGD(net.parameters(), lr=1)
    optimizer = kraft.optim.Adam(net.parameters(), lr=5e-3)

    for epoch in range(20):
        random.shuffle(train)

        train_epoch(net, device, optimizer, regularizer, train)

    net.eval()

    test_epoch(net, device, test)


if __name__ == "__main__":
    main()
