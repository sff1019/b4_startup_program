import sys, os
import numpy as np
import pickle
sys.path.append(os.pardir)

from activation_functions.sigmoid import sigmoid_function
from activation_functions.softmax import softmax_function
from dataset.mnist import load_mnist


def get_data():
    """
    画像を
    784個の要素からなる1次元配列
    0.0~1.0の値に正規化する
    """
    (x_train, t_train), (x_test, t_test) = load_mnist(
            flatten=True, normalize=True, one_hot_label=False)
    return x_test, t_test


def init_network():
    with open('sample_weight.pkl', 'rb') as f:
        network = pickle.load(f)

    return network


def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']


    a1 = np.dot(x, W1) + b1
    z1 = sigmoid_function(a1)

    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid_function(a2)

    a3 = np.dot(z2, W3) + b3
    y = softmax_function(a3)

    return y


if __name__ == '__main__':
    x, t = get_data()
    network = init_network()

    accuracy_cnt = 0

    for i in range(len(x)):
        y = predict(network, x[i])
        p = np.argmax(y)

        if p == t[i]:
            accuracy_cnt += 1

        print('Accuracy:' + str(float(accuracy_cnt) / len(x)))
