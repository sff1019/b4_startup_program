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
# parser.add_argument(
#     '--model_type',
#     choices=['lenet', 'mobilenet', 'mobilenetv2', 'vgg16', 'resnet50'],
#     default='lenet',
#     help='Currently supports LeNet5, MobileNet, MobileNetV2, VGG16, ResNet50'
# )
# parser.add_argument('--device', choices=['cpu', 'gpu'], default='gpu')
# parser.add_argument('--net_type', choices=['mlp', 'cnn'], default='mlp')

args = parser.parse_args()

path_header = 'net_comp_gpu'
# optimizers = ['adam',  'adagrad', 'msgd', 'rmsprop', 'sgd']
model_type = ['lenet', 'mobilenet', 'mobilenetv2', 'vgg16', 'resnet50']

if args.plot_type == 'accuracy':
    data_list = ['main/accuracy', 'test/accuracy']
elif args.plot_type == 'loss':
    data_list = ['main/loss']
elif args.plot_type == 'elapsed_time':
    data_list = ['elapsed_time']


def load_log():
    results = {}
    for model in model_type:
        with open(
            f'./outputs/{path_header}_{model}/log'
        ) as f:
            results[model] = ast.literal_eval(f.read())

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
    for i in range(50):
        header = header + f' {i} |'
        alignment = alignment + f':-:|'

    print(header)
    print(alignment)

    for model in lst:
        row = f'|{model}|'
        for data in data_list:
            for num in models_results[model][data]:
                row += f'{round(float(num), 3)}|'
        print(row)


if __name__ == '__main__':
    max_epochs = 50

    results = load_log()

    models_results = {}
    for model in results:
        models_results[model] = extract_training_data(
            data_list, results[model][0]
        )

    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    print(models_results)
    for model in models_results:
        for data in data_list:
            ax.plot(
                list(range(max_epochs)),
                models_results[model][data],
                label=f'{model}-{data}',
                # marker='x'
            )

    print_table(models_results)
    handles, labels = ax.get_legend_handles_labels()
    lgd = ax.legend(handles, labels, loc='upper left',
                    bbox_to_anchor=(1, 1))
    ax.set_title(
        f'GPU: {args.plot_type.capitalize()} Comparison Based on model')
    ax.set_xlabel('Epoch Num')
    ax.set_ylabel(f'{args.plot_type.capitalize()}')
    ax.grid(True)
    fig.savefig(f'assets/{path_header}_{args.plot_type}.svg',
                bbox_inches='tight')
