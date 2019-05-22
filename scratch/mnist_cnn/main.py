import numpy as np
import matplotlib.pyplot as plt

from adam import Adam
from load_data import load_data
from net import Net


if __name__ == '__main__':
    x_train, t_train, x_test, t_test = load_data()

    conv_param = {
        'filter_num': 30,
        'filter_size': 5,
        'pad': 0,
        'stride': 1,
    }
    hidden_size = 256
    output_size = 10

    network = Net(
        input_dim=(1, 28, 28),
        conv_param=conv_param,
        hidden_size=hidden_size,
        output_size=output_size,
        weight_init_std=0.01,
    )
    optimizer = Adam()

    train_size = x_train.shape[0]
    batch_size = 64

    learning_rate = 0.1
    # epochs = 100

    train_loss_list = []
    train_acc_list = []
    test_acc_list = []

    epochs = 20
    iter_per_epoch = max(train_size / batch_size, 1)
    max_iter = int(epochs * iter_per_epoch)

    for i in range(max_iter):
        batch_mask = np.random.choice(train_size, batch_size)

        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]

        grad = network.gradient(x_batch, t_batch)
        optimizer.update(network.params, grad)

        loss = network.loss(x_batch, t_batch)
        train_loss_list.append(loss)
        print(f'train loss: {loss}')

        for key in ('W1', 'b1', 'W2', 'b2', 'W3', 'b3'):
            network.params[key] -= learning_rate * grad[key]

        loss = network.loss(x_batch, t_batch)
        train_loss_list.append(loss)

        if i % iter_per_epoch == 0:
            train_acc = network.accuracy(x_train, t_train)
            test_acc = network.accuracy(x_test, t_test)
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)
            print(train_acc, test_acc)

    markers = {'train': 'o', 'test': 's'}
    x = np.arange(epochs)
    plt.plot(x, train_acc_list, marker='o', label='train', markevery=2)
    plt.plot(x, test_acc_list, marker='s', label='test', markevery=2)
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.ylim(0, 1.0)
    plt.legend(loc='lower right')
    plt.show()
