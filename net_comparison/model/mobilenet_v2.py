import torch.nn as nn


class Bottleneck(nn.Module):
    def __init__(self, in_dim, out_dim, stride, expand_ratio):
        super(Bottleneck, self).__init__()
        self.stride = stride
        self.in_dim = in_dim
        self.out_dim = out_dim

        hidden_dim = in_dim * expand_ratio

        # pointwise convolution
        self.pw = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, kernel_size=1,
                      stride=1, padding=0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
        )

        # depthwise convolution
        self.dw = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3,
                      stride=stride, padding=1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
        )

        # pointwize linear
        self.pw_linear = nn.Sequential(
            nn.Conv2d(hidden_dim, out_dim,
                      kernel_size=1, bias=False),
            nn.BatchNorm2d(out_dim),
        )

    def forward(self, x):
        out = self.pw(x)
        out = self.dw(out)
        out = self.pw_linear(out)

        if self.stride == 1 and self.in_dim == self.out_dim:
            out += x

        return out


class MobileNetV2(nn.Module):
    def __init__(self):
        super(MobileNetV2, self).__init__()

        self.conv_ir = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),

            Bottleneck(in_dim=32, out_dim=16, stride=1, expand_ratio=1),

            Bottleneck(in_dim=16, out_dim=24, stride=1, expand_ratio=6),
            Bottleneck(in_dim=24, out_dim=24, stride=1, expand_ratio=6),

            Bottleneck(in_dim=24, out_dim=32, stride=2, expand_ratio=6),
            Bottleneck(in_dim=32, out_dim=32, stride=1, expand_ratio=6),
            Bottleneck(in_dim=32, out_dim=32, stride=1, expand_ratio=6),

            Bottleneck(in_dim=32, out_dim=64, stride=2, expand_ratio=6),
            Bottleneck(in_dim=64, out_dim=64, stride=1, expand_ratio=6),
            Bottleneck(in_dim=64, out_dim=64, stride=1, expand_ratio=6),
            Bottleneck(in_dim=64, out_dim=64, stride=1, expand_ratio=6),

            Bottleneck(in_dim=64, out_dim=96, stride=1, expand_ratio=6),
            Bottleneck(in_dim=96, out_dim=96, stride=1, expand_ratio=6),
            Bottleneck(in_dim=96, out_dim=96, stride=1, expand_ratio=6),

            Bottleneck(in_dim=96, out_dim=160, stride=2, expand_ratio=6),
            Bottleneck(in_dim=160, out_dim=160, stride=1, expand_ratio=6),
            Bottleneck(in_dim=160, out_dim=160, stride=1, expand_ratio=6),

            Bottleneck(in_dim=160, out_dim=320, stride=1, expand_ratio=6),

            nn.Conv2d(in_channels=320, out_channels=1280,
                      kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(1280),
            nn.ReLU6(inplace=True),

            nn.AvgPool2d(7),
        )

        self.fc = nn.Linear(1280, 10)

    def forward(self, x):
        out = self.conv_ir(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out
