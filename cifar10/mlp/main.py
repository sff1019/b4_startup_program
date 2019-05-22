import argparse
import matplotlib
import os
import sys
import time
import torch.nn as nn
# import torch.optim as optim
sys.path.append(os.pardir)

try:
    from dataset import load
    from evaluator import evaluate_classes
    from log import plot_log
    from model import MLP
    from optimizers import select_optimizer
    from trainer import train
except ImportError as e:
    print(e)


output_file = 'cifar10_sgd_result'
opt_type = sys.argv[1]

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='DNI')
    parser.add_argument(
        '--optimizer_type',
        choices=['sgd', 'msgd', 'adagrad', 'rmsprop', 'adam'],
        default='sgd',
        help='currently supports SGD, Momentum SGD, AdaGrad, RMSprop and Adam'
    )
    parser.add_argument('--device', choices=['cpu', 'gpu'], default='cpu')

    args = parser.parse_args()

    batch_size = 128
    epochs = 10

    # load datasets
    train_loader, test_loader, classes = load()

    # Calculate computing time
    start_time = time.time()

    # Set model
    model = MLP()

    optimizer = select_optimizer(args.optimizer_type, model)

    loss_fn = nn.CrossEntropyLoss()

    # Train data
    train_result = train(
        epochs,
        loss_fn,
        model,
        optimizer,
        start_time,
        test_loader,
        train_loader,
    )

    classified_result = evaluate_classes(test_loader, model, classes)

    plot_log([train_result, classified_result],
             f'cifar_mlp_{args.device}_{args.optimizer_type}')
