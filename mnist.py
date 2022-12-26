import kraft
import kraft.device
import kraft.optim
import kraft.autograd.ops
from kraft import nn


class MnistConv(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4),
            nn.ReLU(),
        )

        self.fc = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
            nn.ReLU(),
        )

        self.softmax = nn.Softmax()

    def forward(self, xs):
        return xs


def main():
    device = kraft.device.get_gpu_device()




if __name__ == "__main__":
    main()
