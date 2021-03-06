"""
This function is to compare accuracy based on optimizers
"""
import argparse
import ast
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='DNI')
parser.add_argument(
    '--plot_type',
    choices=['accuracy', 'loss', 'elapsed_time'],
    default='accuracy',
    help='currently supports Accuracy, Loss and Evaluated Time'
)
parser.add_argument('--device', choices=['cpu', 'gpu'], default='gpu')
parser.add_argument('--net_type', choices=['mlp', 'cnn'], default='mlp')
parser.add_argument(
    '--dir',
    choices=['outputs_e10_b128', 'outputs_e50_b4'],
    default='outputs_e50_b4'
)
parser.add_argument('--epochs', choices=['10', '50'], default='50')

args = parser.parse_args()

path_header = 'cifar10_'
optimizers = ['adam',  'adagrad', 'msgd', 'rmsprop', 'sgd']

if args.plot_type == 'accuracy':
    data_list = ['main/accuracy', 'test/accuracy']
elif args.plot_type == 'loss':
    data_list = ['main/loss']
elif args.plot_type == 'elapsed_time':
    data_list = ['elapsed_time']


def load_log():
    results = {}
    for optimizer in optimizers:
        with open(
            f'./outputs/{args.dir}/{path_header}{args.net_type}_{args.device}_{optimizer}/log'
        ) as f:
            results[optimizer] = ast.literal_eval(f.read())

    return results


def extract_training_data(fields, lst):
    extracted = {}
    for field in fields:
        field_lst = []
        for item in lst:
            field_lst.append(item[field])
        extracted[field] = field_lst

    return extracted


def print_table(lst):
    header = '||'
    alignment = '|-----------|'
    for i in range(int(args.epochs)):
        header = header + f' {i} |'
        alignment = alignment + f':-:|'

    # print('|| 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10|  ')
    # print('|--------------|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|  ')
    print(header)
    print(alignment)

    for optimizer in lst:
        row = f'|{optimizer}|'
        for data in data_list:
            for num in optimizers_results[optimizer][data]:
                row += f'{round(float(num), 3)}|'
        print(row)


if __name__ == '__main__':
    max_epochs = int(args.epochs)

    results = load_log()

    # print(results)
    #
    optimizers_results = {}
    for optimizer in results:
        optimizers_results[optimizer] = extract_training_data(
            data_list, results[optimizer][0]
        )

    fig = plt.figure(1)
    fig.suptitle(
        f'{args.device.upper()}: {args.plot_type.capitalize()} Comparison Based on Optimizer')
    ax = fig.add_subplot(111)
    print(optimizers_results)
    for optimizer in optimizers_results:
        for data in data_list:
            ax.plot(
                list(range(max_epochs)),
                optimizers_results[optimizer][data],
                label=f'{optimizer}-{data}',
                # marker='x'
            )

    print_table(optimizers_results)
    handles, labels = ax.get_legend_handles_labels()
    lgd = ax.legend(handles, labels, loc='upper left',
                    bbox_to_anchor=(1, 1))
    ax.set_title(
        f'Epochs: {max_epochs}')
    ax.set_xlabel('Epoch Num')
    ax.set_ylabel(f'{args.plot_type.capitalize()}')
    ax.grid(True)
    fig.savefig(f'assets/{path_header}{args.net_type}_{args.device}_{args.epochs}_{args.plot_type}.svg',
                bbox_inches='tight')
