import numpy as np


def sigmoid_function(x):
    return 1 / (1 + np.exp(-x))

