import numpy as np
from collections import OrderedDict

from affine import Affine
from relu import Relu
from softmax import Softmax


class Net:
    def __init__(self, input_size, hidden_size, output_size,
                 weight_init_std=0.01):

        self.params = {}

        # Weights and biases
        self.params['W1'] = weight_init_std * \
            np.random.rand(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)

        self.params['W2'] = weight_init_std * \
            np.random.rand(hidden_size, hidden_size)
        self.params['b2'] = np.zeros(hidden_size)

        self.params['W3'] = weight_init_std * \
            np.random.rand(hidden_size, output_size)
        self.params['b3'] = np.zeros(output_size)

        # Layers
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(
            self.params['W1'], self.params['b1']
        )
        self.layers['Relu1'] = Relu()

        self.layers['Affine2'] = Affine(
            self.params['W2'], self.params['b2']
        )
        self.layers['Relu2'] = Relu()

        self.layers['Affine3'] = Affine(
            self.params['W3'], self.params['b3']
        )
        self.lastLayer = Softmax()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)

        return x

    def loss(self, x, t):
        y = self.predict(x)

        return self.lastLayer.forward(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1:
            t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])

        return accuracy

    def gradient(self, x, t):
        self.loss(x, t)

        dout = 1
        dout = self.lastLayer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        grads = {}
        for index in range(int(len(layers) / 2) + 1):
            grads[f'W{index+1}'] = self.layers[f'Affine{index+1}'].dW
            grads[f'b{index+1}'] = self.layers[f'Affine{index+1}'].db

        return grads
