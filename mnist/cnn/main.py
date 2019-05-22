from chainer import iterators
from chainer import optimizers
from chainer import training
from chainer.training import extensions
import chainer.links as L
import matplotlib
matplotlib.use('Agg')

try:
    from dataset import load, split_train
    from model import LeNet5, CNN
except ImportError:
    print('Error while importing local files occured')


output_file = 'mnist_rmsprop_result'

if __name__ == '__main__':
    batch_size = 128
    max_epochs = 10

    # load datasets
    train_val, test = load()

    # split the training dataset to train and validation
    train, valid = split_train(train_val)

    # create iterations
    train_iter = iterators.SerialIterator(train, batch_size)
    valid_iter = iterators.SerialIterator(
        valid, batch_size, repeat=False, shuffle=False)
    test_iter = iterators.SerialIterator(
        test, batch_size, repeat=False, shuffle=False)

    # model = LeNet5()
    model = CNN()
    model = L.Classifier(model)
    device = -1

    if device >= 0:
        model.to_gpu(device)

    optimizer = optimizers.RMSprop()

    optimizer.setup(model)

    updater = training.updaters.StandardUpdater(
        train_iter, optimizer, device=device)

    trainer = training.Trainer(
        updater, (max_epochs, 'epoch'), out=output_file
    )

    # add extensions to current trainer
    trainer.extend(extensions.LogReport())  # save log fiels automatically
    # automatically serialize the state periodically
    trainer.extend(extensions.snapshot(
        filename='snapshot_epoch--{.updater.epoch}'))
    # evaluate models on validation set
    trainer.extend(extensions.Evaluator(test_iter, model, device=device))
    # print states
    trainer.extend(extensions.PrintReport([
        'epoch',
        'main/loss',
        'main/accuracy',
        'validation/main/loss',
        'validation/main/accuracy',
        'elapsed_time',
    ]))
    trainer.extend(extensions.PlotReport(
        ['main/accuracy', 'validation/main/accuracy'],
        x_key='epoch',
        file_name='accuracy.png'
    ))
    trainer.extend(extensions.dump_graph('main/loss'))

    trainer.run()
