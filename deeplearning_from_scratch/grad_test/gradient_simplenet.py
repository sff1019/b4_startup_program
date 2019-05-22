import sys, os
import numpy as np
sys.path.append(os.pardir)

from activation_functions.softmax import softmax_function
from loss_functions.cross_entropy_error import cross_entropy_error_function
from differentiation_functions.gradient import numerical_gradient_function


class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2, 3)


    def predict(self, x):
        return np.dot(x, self.W)


    def loss(self, x, t):
        z = self.predict(x)
        y = softmax_function(z)
        loss = cross_entropy_error_function(y, t)

        return loss
