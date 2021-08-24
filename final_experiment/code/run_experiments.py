#!/usr/bin/env python

import subprocess
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', type=str, default=None, choices=['MLP','CNN','GraphConv','LR'])
parser.add_argument('-ws', '--window_size', type=float, default=None, choices=[0.5, 1, 1.5, 2])
parser.add_argument('-ef', '--eeg_feature', type=str, default=None, choices=['wav','psd','raw'])
args = parser.parse_args()

models = [args.model] if args.model else ['MLP','CNN','GraphConv','LR']
window_sizes = [args.window_size] if args.window_size else [0.5, 1, 1.5, 2]
eeg_features = [args.eeg_feature] if args.eeg_feature else ['wav','psd','raw']

lr = 5e-4
l2 = 0 
dr = 0.25

hidden_channels = [64, 128, 256, 512, 1024, 2048]
number_train_samples = [1, 2, 4, 8]

if models[0] == 'LR':
	hidden_channels = [64]


grid_search = [(model,ws,ef,hc,nts) for model in models for ws in window_sizes for nts in number_train_samples for ef in eeg_features for hc in hidden_channels ]
total_runs = len(grid_search) 


print(f'Running {total_runs} experiments')
current_run = 1
for model,ws,ef,hc,nts in grid_search:

	if model == 'LR' and hc != 64:
		continue

	test_model_dict = f'{model}_{ws}_{ef}'
	bashCommand = f"python3 main.py -ws {ws} -ef {ef} -hc {hc} -nts {nts} -wd {l2} -dr {dr} -m {model} -lr {lr} -wtr -esp 30 -trd {test_model_dict} -tmd {test_model_dict}"
	print(f'Running ({current_run} / {total_runs}): {bashCommand}')
	process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
	output, error = process.communicate()
	print('-OK-')
	current_run+=1
