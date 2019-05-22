import numpy as np
import matplotlib.pylab as plt


def function_1(x):
    return x[0] ** 2 + x[1] ** 2


def numerical_gradient_function(f, x):
    h = 1e-4
    grad = np.zeros_like(x)

    for idx in range(x.size):
        tmp_val = x[idx]

        x[idx] = tmp_val + h
        fxh1 = f(x)

        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val

    return grad


def gradient_descent_function(f, init_x, lr=0.01, step_num=100):
    x = init_x
    x_history = []

    for i in range(step_num):
        x_history.append(x.copy())
        grad = numerical_gradient_function(f, x)

        x -= lr * grad

    return x, np.array(x_history)


if __name__ == '__main__':
    init_x = np.array([-3.0, 4.0])
    x, plt_x = gradient_descent_function(function_1, init_x=init_x, lr=0.1, step_num=100)

    plt.plot( [-5, 5], [0,0], '--b')
    plt.plot( [0,0], [-5, 5], '--b')
    plt.plot(plt_x[:,0], plt_x[:,1], 'o')

    plt.xlim(-3.5, 3.5)
    plt.ylim(-4.5, 4.5)
    plt.xlabel("X0")
    plt.ylabel("X1")
    plt.show()
