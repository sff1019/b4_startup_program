import torch
import torch.nn as nn
import torch.nn.functional as F

IMG_WIDTH = 32
IMG_HEIGHT = 32
COL_CHAN = 3


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(IMG_WIDTH * IMG_HEIGHT * COL_CHAN, 100),
            nn.ReLU(),
            nn.Linear(100, 10),
        )

    def forward(self, x):
        x = x.view(-1, IMG_WIDTH * IMG_HEIGHT * COL_CHAN)
        x = self.layers(x)

        return x
