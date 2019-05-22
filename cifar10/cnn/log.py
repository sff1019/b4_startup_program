import os
import json


def plot_log(item, out):
    """
    Dump results data into log file
    """
    output_path = f'{out}/log'

    if not os.path.exists(out):
        os.makedirs(out)

    with open(output_path, 'w+') as f:
        json.dump(item, f, indent=2)
