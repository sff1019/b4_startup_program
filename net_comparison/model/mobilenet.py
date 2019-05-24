import torch.nn as nn


IMG_WIDTH = 32
IMG_HEIGHT = 32
COLOR_CHANNEL = 3


class DepthwiseBlock(nn.Module):
    def __init__(self, in_channels, kernel_size, stride=1, padding=1, bias=False):
        super(DepthwiseBlock, self).__init__()
        self.dw_block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride,
                      padding=padding, groups=in_channels, bias=bias),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        out = self.dw_block(x)

        return out


class OneByOneBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, padding=1, bias=False):
        super(OneByOneBlock, self).__init__()
        self.one_by_one_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1,
                      stride=stride, padding=padding, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        out = self.one_by_one_block(x)

        return out


class MobileNet(nn.Module):
    def __init__(self):
        super(MobileNet, self).__init__()

        self.conv_bn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            DepthwiseBlock(32, kernel_size=3),
            OneByOneBlock(32, 64),

            DepthwiseBlock(64, kernel_size=3, stride=2),
            OneByOneBlock(64, 128),

            DepthwiseBlock(128, kernel_size=3),
            OneByOneBlock(128, 128),

            DepthwiseBlock(128, kernel_size=3, stride=2),
            OneByOneBlock(128, 256),

            DepthwiseBlock(256, kernel_size=3),
            OneByOneBlock(256, 256),

            DepthwiseBlock(256, kernel_size=3, stride=2),
            OneByOneBlock(256, 512),

            DepthwiseBlock(512, kernel_size=3),
            OneByOneBlock(512, 512),
            DepthwiseBlock(512, kernel_size=3),
            OneByOneBlock(512, 512),
            DepthwiseBlock(512, kernel_size=3),
            OneByOneBlock(512, 512),
            DepthwiseBlock(512, kernel_size=3),
            OneByOneBlock(512, 512),
            DepthwiseBlock(512, kernel_size=3),
            OneByOneBlock(512, 512),

            DepthwiseBlock(512, kernel_size=3),
            OneByOneBlock(512, 1024),

            DepthwiseBlock(1024, kernel_size=3, stride=2),
            OneByOneBlock(1024, 1024),
            nn.AvgPool2d(7),
        )

        self.fc = nn.Linear(1024, 1000)

    def forward(self, x):
        out = self.conv_bn(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out
