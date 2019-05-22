"""
This function is to compare accuracy based on optimizers
"""
import argparse
import ast
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='DNI')
parser.add_argument(
    '--plot_type',
    choices=['accuracy', 'loss', 'evaluated_time'],
    default='accuracy',
    help='currently supports Accuracy, Loss and Evaluated Time'
)
parser.add_argument('--device', choices=['cpu', 'gpu'], default='cpu')
parser.add_argument('--net_type', choices=['mlp', 'cnn'], default='mlp')

args = parser.parse_args()

# # device = 'cpu'
# device = 'gpu'
path_header = 'cifar10_'
optimizers = ['adam',  'adagrad', 'momentumsgd', 'rmsprop', 'sgd']
# optimizers = ['adam',  'adagrad', 'momentum_sgd', 'rmsprop', 'sgd']
path_tail = '_result'
data_list = ['main/accuracy', 'validation/main/accuracy']


def load_log():
    results = {}
    for optimizer in optimizers:
        with open(
            f'./{args.net_type}/{args.device}/{path_header}{optimizer}{path_tail}/log'
        ) as f:
            results[optimizer] = ast.literal_eval(f.read())

    return results


def extract_data(fields, lst):
    extracted = {}
    for field in fields:
        field_lst = []
        for item in lst:
            field_lst.append(item[field])
        extracted[field] = field_lst

    return extracted


# def print_table(lst):
#     print('|| 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10|  ')
#     print('|--------------|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|  ')
#
#     for optimizer in lst:
#         row = f'|{optimizer}|'
#         for data in data_list:
#             for num in optimizers_results[optimizer][data]:
#                 row += f'{round(float(num), 3)}|'
#         print(row)


if __name__ == '__main__':
    max_epochs = 10

    results = load_log()

    print(results)
    #
    # optimizers_results = {}
    # for optimizer in results:
    #     optimizers_results[optimizer] = extract_data(
    #         data_list, results[optimizer]
    #     )
    #
    # for optimizer in optimizers_results:
    #     for data in data_list:
    #         plt.plot(
    #             list(range(max_epochs)),
    #             optimizers_results[optimizer][data],
    #             label=f'{optimizer}-{data}',
    #             marker='x'
    #         )
    #
    # print_table(optimizers_results)
    # plt.legend()
    # plt.xlabel('Epoch Num')
    # plt.ylabel(f'{args.plot_type}')
    # plt.title(f'{args.device.upper()}: Accuracy Comparison Based on Optimizer')
    # plt.grid(True)
    # plt.savefig(f'../assets/mlp_{args.device}_comp_acc.png')
