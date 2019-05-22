from torch import optim


def select_optimizer(opt, model):
    """
    Return selected optimizers.
    Default value is SGD
    """

    if opt == 'msgd':
        return optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    elif opt == 'adam':
        return optim.Adam(model.parameters(), lr=0.001)
    elif opt == 'rmsprop':
        return optim.RMSprop(model.parameters(), lr=0.001)
    elif opt == 'adagrad':
        return optim.Adagrad(model.parameters(), lr=0.001)
    else:
        return optim.SGD(model.parameters(), lr=0.001)
