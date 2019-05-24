import torch.nn as nn


class ConvBlock(nn.Module):
    expansion = 4

    def __init__(self, in_dim, out_dim, stride=1):
        super(ConvBlock, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(out_dim, out_dim, kernel_size=3,
                      stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(out_dim, out_dim * self.expansion,
                      kernel_size=1, bias=False),
            nn.BatchNorm2d(out_dim * self.expansion),
        )

        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_dim != out_dim * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_dim, out_dim * self.expansion, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_dim * self.expansion),
            )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)

        out += self.shortcut(x)

        out = self.relu(out)

        return out


class ResNet50(nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()
        self.in_dim = 64

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.layer2 = self._make_layer(ConvBlock, 64, 3, stride=1)
        self.layer3 = self._make_layer(ConvBlock, 128, 4, stride=2)
        self.layer4 = self._make_layer(ConvBlock, 256, 6, stride=2)
        self.layer5 = self._make_layer(ConvBlock, 512, 3, stride=2)

        self.avg_pool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * ConvBlock.expansion, 10)

    def _make_layer(self, block, out_dim, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_dim, out_dim, stride))
            self.in_dim = out_dim * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)

        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out
