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


class MnistConv(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=7),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4),
            nn.ReLU(),
        )

        self.fc = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(p=0.25),
            nn.Linear(128, 10),
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

    inputs = kraft.Variable(np.concatenate(inputs, axis=0), device=device)
    target = kraft.Variable(np.concatenate(targets, axis=0), device=device)

    return inputs, target


def train_epoch(net, device, optimizer, regularizer, dataset):
    for chunk in chunked(tqdm(dataset), 256):
        optimizer.zero_grad()

        inputs, target = inputs_targets_from_chunk(chunk, device)

        outputs = net(inputs)

        loss = fun.ce_loss(outputs, target, reduction="mean")
        loss = regularizer.add_to_loss(loss)

        loss.backward()
        optimizer.step()


def test_epoch(net, device, dataset):
    correct_answers = 0

    answers = Counter()

    for sample, label in tqdm(dataset):
        inputs = kraft.Variable(np.array(sample, dtype=np.float32) / 255, device=device)
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
    regularizer = nn.L2Regularizer(net.parameters(), alpha=1e-1, reduction="mean")

    optimizer = kraft.optim.Adam(net.parameters(), lr=5e-3)

    for epoch in range(20):
        random.shuffle(train)

        train_epoch(net, device, optimizer, regularizer, train)

    net.eval()

    test_epoch(net, device, test)


if __name__ == "__main__":
    main()
