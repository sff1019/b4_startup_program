import argparse
import torch
from torch import optim
import torch.nn as nn

from evaluator import evaluate_classes
from dataset import cifar10_load
from model.selector import selector
from report import plot_log
from trainer import Trainer


parser = argparse.ArgumentParser(description='DNI')
parser.add_argument(
    '--model_type',
    choices=['lenet', 'mobilenet', 'mobilenetv2', 'vgg16', 'resnet50'],
    default='lenet',
    help='Currently supports LeNet5, MobileNet, MobileNetV2, VGG16, ResNet50'
)
parser.add_argument('--device', choices=['cpu', 'gpu'], default='cpu')

args = parser.parse_args()


if __name__ == '__main__':
    batch_size = 4
    epochs = 1

    train_loader, test_loader, classes = cifar10_load()

    # Set model
    model = selector(args.model_type)

    # Set device
    if args.device == 'gpu':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cuda')
        model.to(device)
    else:
        device = 'cpu'

    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()

    trainer = Trainer(criterion, device, model,
                      optimizer, test_loader, train_loader, epochs)

    trainer.run()
    torch.save(model.state_dict(), f'model/trained/{args.model_type}.pkl')
    classified_results = evaluate_classes(classes, device, test_loader, model)

    plot_log([trainer.train_results, classified_results],
             f'net_comp_{args.device}_{args.model_type}')
