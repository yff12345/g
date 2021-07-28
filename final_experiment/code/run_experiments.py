#!/usr/bin/env python

import subprocess
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', type=str, default='MLP', choices=['MLP','CNN','GraphConv','GIN'], required=True)
parser.add_argument('-ws', '--window_size', type=float, default=None)
parser.add_argument('-tmd', '--test_model_dict', type=str, default=None, help='Model to test', required=True)
parser.add_argument('-trd', '--test_results_dir', type=str, default=None, required = True)
args = parser.parse_args()

model = args.model
lr = 5e-3 if model in ['MLP'] else 5e-4
window_sizes = [args.window_size] if args.window_size else [0.5, 1, 1.5, 2]
eeg_features = ['wav','psd','raw']
hidden_channels = [32 ,64, 128]
number_train_samples = [1, 2, 4, 8]
l2s = [0,0.01,0.04,0.16, 0.64]
drs = [0,0.25,0.5]


grid_search = [(ws,ef,hc,nts,l2,dr) for ws in window_sizes for nts in number_train_samples for ef in eeg_features for hc in hidden_channels for l2 in l2s for dr in drs ]
total_runs = len(grid_search) 


print(f'Running {total_runs} experiments')
current_run = 1
for ws,ef,hc,nts,l2,dr in grid_search:
	bashCommand = f"python3 main.py -ws {ws} -ef {ef} -hc {hc} -nts {nts} -wd {l2} -dr {dr} -m {model} -lr {lr} -wtr -esp 30 -trd {args.test_results_dir} -tmd {args.test_model_dict}"
	print(f'Running ({current_run} / {total_runs}): {bashCommand}')
	process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
	output, error = process.communicate()
	print('-OK-')
	current_run+=1
