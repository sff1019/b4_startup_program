import sys
import os
import numpy as np
sys.path.append(os.pardir)

from activation_functions.sigmoid import sigmoid_function
from activation_functions.softmax import softmax_function
from differentiation_functions.gradient import numerical_gradient_function
from loss_functions.cross_entropy_error import cross_entropy_error_function


class TwoLayerNet(object):

    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.params = {}
        self.params['W1'] = weight_init_std * \
            np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * \
            np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']

        a1 = np.dot(x, W1) + b1
        z1 = sigmoid_function(a1)

        a2 = np.dot(z1, W2) + b2
        y = softmax_function(a2)

        return y

    def loss(self, x, t):
        y = self.predict(x)

        return cross_entropy_error_function(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def numerical_gradient(self, x, t):
        def loss_W(W): return self.loss(x, t)

        grads = {}
        grads['W1'] = numerical_gradient_function(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient_function(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient_function(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient_function(loss_W, self.params['b2'])

        return grads
