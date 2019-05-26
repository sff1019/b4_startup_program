import os
import json


def plot_log(item, out):
    """
    Dump results data into log file
    """
    output_path = f'outputs/{out}/log'

    if not os.path.exists(f'outputs/{out}'):
        os.makedirs(f'outputs/{out}')

    with open(output_path, 'w+') as f:
        json.dump(item, f, indent=2)
