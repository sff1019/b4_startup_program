import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision import transforms


def cifar10_load(withlabel=True, ndim=3):
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(
             mean=(0.49139968, 0.48215841, 0.44653091),
             std=(0.24703223, 0.24348513, 0.26158784)
        )])

    train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                             download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=4,
                                               shuffle=True, num_workers=2)

    test_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=4,
                                              shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')

    return train_loader, test_loader, classes


def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
