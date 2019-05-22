import chainer
from chainer import Chain
import chainer.links as L
import chainer.functions as F


class LeNet5(Chain):
    def __init__(self):
        super(LeNet5, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(
                in_channels=1, out_channels=6, ksize=5, stride=1
            )
            self.conv2 = L.Convolution2D(
                in_channels=6, out_channels=16, ksize=5, stride=1
            )
            self.conv3 = L.Convolution2D(
                in_channels=16, out_channels=120, ksize=4, stride=1
            )
            self.fc4 = L.Linear(None, 84)
            self.fc5 = L.Linear(84, 10)

    def forward(self, x):
        h = F.sigmoid(self.conv1(x.reshape((-1, 1, 28, 28))))
        h = F.max_pooling_2d(h, ksize=2, stride=2)
        h = F.sigmoid(self.conv2(h))
        h = F.max_pooling_2d(h, ksize=2, stride=2)
        h = F.sigmoid(self.conv3(h))
        h = F.sigmoid(self.fc4(h))

        if chainer.config.train:
            return self.fc5(h)
        # return F.softmax(self.fc5(h))
        return self.fc5(h)


class CNN(Chain):
    def __init__(self, train=True):
        super(CNN, self).__init__(
            conv1=L.Convolution2D(1, 64, 5),
            conv2=L.Convolution2D(64, 128, 5),
            l1=L.Linear(2048, 10),
        )
        self.train = train

    def forward(self, x):
        h = F.relu(self.conv1(x))
        h = F.max_pooling_2d(h, 2)
        h = F.relu(self.conv2(h))
        h = F.max_pooling_2d(h, 2)
        return self.l1(h)
