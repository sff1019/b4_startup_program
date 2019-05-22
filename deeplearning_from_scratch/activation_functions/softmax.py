import numpy as np


def softmax_function(x):
    c = np.max(x)
    exp_x = np.exp(x - c)

    sum_exp_x = np.sum(exp_x)

    return exp_x / sum_exp_x
