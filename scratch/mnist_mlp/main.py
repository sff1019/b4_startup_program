import numpy as np

from load_data import load_data
from net import Net


if __name__ == '__main__':
    x_train, t_train, x_test, t_test = load_data()

    input_size = 784
    hidden_size = 256
    output_size = 10

    network = Net(input_size, hidden_size, output_size)

    train_size = x_train.shape[0]
    batch_size = 64

    learning_rate = 0.1
    epochs = 10000

    train_loss_list = []
    train_acc_list = []
    test_acc_list = []

    iter_per_epoch = max(train_size / batch_size, 1)

    for i in range(epochs):
        batch_mask = np.random.choice(train_size, batch_size)
        print(batch_mask)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]

        grad = network.gradient(x_batch, t_batch)

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
