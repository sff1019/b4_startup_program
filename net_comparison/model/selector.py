from model.lenet import LeNet5
from model.mobilenet import MobileNet
from model.mobilenet_v2 import MobileNetV2
from model.resnet50 import ResNet50
from model.vgg16 import VGG16


def selector(args):
    if args == 'lenet':
        return LeNet5()
    elif args == 'mobilenet':
        return MobileNet()
    elif args == 'mobilenetv2':
        return MobileNetV2()
    elif args == 'vgg16':
        return VGG16()
    elif args == 'resnet50':
        return ResNet50()
