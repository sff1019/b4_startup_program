import torch.nn as nn


class VGGConvolution(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(VGGConvolution, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1),
            # nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        out = self.conv(x)

        return out


class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()

        self.layer = nn.Sequential(
            VGGConvolution(3, 64),
            VGGConvolution(64, 64),
            nn.MaxPool2d(kernel_size=2, stride=2),

            VGGConvolution(64, 128),
            VGGConvolution(128, 128),
            nn.MaxPool2d(kernel_size=2, stride=2),

            VGGConvolution(128, 256),
            VGGConvolution(256, 256),
            VGGConvolution(256, 256),
            nn.MaxPool2d(kernel_size=2, stride=2),

            VGGConvolution(256, 512),
            VGGConvolution(512, 512),
            VGGConvolution(512, 512),
            nn.MaxPool2d(kernel_size=2, stride=2),

            VGGConvolution(512, 512),
            VGGConvolution(512, 512),
            VGGConvolution(512, 512),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.avg_pool = nn.AvgPool2d(kernel_size=1, stride=1)
        self.fc = nn.Linear(512, 10)

    def forward(self, x):
        out = self.layer(x)
        out = self.avg_pool()
        out = out.view(out.size(0), -1)

        out = self.fc(out)

        return out
