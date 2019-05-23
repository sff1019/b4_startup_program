import torch
import torch.nn as nn
import torch.nn.functional as F

IMG_WIDTH = 32
IMG_HEIGHT = 32
COL_CHAN = 3


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.layer3 = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.Linear(120, 84),
            nn.Linear(84, 10),
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)

        x = x.view(-1, 16*5*5)
        x = self.layer3(x)

        return x
