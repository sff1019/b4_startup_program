import sys
import os
sys.path.append(os.pardir)

try:
    from dataset.mnist import load_mnist
except ImportError:
    print('ERROR IMPORTING')


def load_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)

    return x_train, t_train, x_test, t_test
