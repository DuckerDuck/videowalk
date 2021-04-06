import sys
import numpy as np
from pathlib import Path
from collections import defaultdict
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from pylatex import Tabular

split_to_size = { 0: 4, 1: 1002, 2: 2001, 3: 3001, 4: 4000 }

def read_splits(location, splits=5):
    split_data = defaultdict(list)
    for split in range(splits):
        for seed in [123, 203, 403, 985]:
            result_folder  = Path(f'{location}{split}_{seed}/')
            global_csv = result_folder / Path('global_results-val.csv')
            # sequence_csv = result_folder / Path('per-sequence_results-val.csv')

            if not global_csv.is_file():
                print(f'{location}: Missing split {split}, {seed}')
                continue

            with open(global_csv) as f:
                output = f.readlines()
                values = [float(val) for val in output[1].strip().split(',')]
                split_data[split].append(values)

    headers = ['Videos'] + [o + ' (SD)'  for o in output[0].strip().split(',')]
    return split_data, headers



def read_data():
    split_datas = []
    
    for location in ['./davis_outputs/davis_output_converted_split_', './davis_output_saliency_converted_split_']:
        split_data, headers = read_splits(location, splits=5)
        split_datas.append(split_data)
    
    table = Tabular('l' * len(headers))
    table.add_row(headers)
    table.add_hline()
    
    plt.figure(figsize=(12, 8))
    x = split_to_size.values()
    plot_label = 0 # J&F Mean

    for split_data in split_datas:
        y = []
        y_err = []
        for split, values in split_data.items():
            values = np.stack(values) * 100
            mean = np.mean(values, axis=0)
            y.append(mean[plot_label])
            sd = np.std(values, axis=0)
            y_err.append(sd[plot_label])
            nvideos = split_to_size[split]
            table.add_row([nvideos] + [ f'{m:.3f}({sd:.3f})' for m, sd in zip(mean, sd)])
        table.add_hline()
        plt.errorbar(x, y, yerr=y_err, label=headers[plot_label])
        
    # Baseline
    x_b = np.linspace(0, list(x)[-1], 100)
    y_b = np.zeros(100) + 67.4
    plt.plot(x_b, y_b, linestyle='dashed')
    plt.legend(['Pretrained', 'CRW', 'CRW + Additive Saliency'])
    plt.savefig('crw-data-plot.png')
    
    table.generate_tex('crw-data-table')

if __name__ == '__main__':
    data = read_data()
