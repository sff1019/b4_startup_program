from chainer.datasets import mnist, split_dataset_random


def load():
    train_val, test = mnist.get_mnist(ndim=3)

    return train_val, test


def split_train(train_val, num=50000, seed=0):
    train, valid = split_dataset_random(train_val, num, seed)

    return train, valid
