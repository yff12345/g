#!/usr/bin/env python

import subprocess

window_sizes = [0.25, 0.5, 1, 2, 4]
eeg_features = ['wav','psd']
models = ['CNN','MLP','GraphConv','GIN']
hidden_channels = [8, 16, 32 ,64, 128]
number_train_samples = [1, 2, 4, 8 , 16, 32, 64, 128 , 256]
batch_sizes = [4, 8 , 16]
learning_rates = [0.001, 0.0001, 0.01]
dropout_rates = [0, 0.125,0.25, 0.5]
activations = ['relu', 'tanh']


total_runs = len(window_sizes) * len(eeg_features) * len(models) * len(hidden_channels) * len(number_train_samples) * len(batch_sizes)  * len(learning_rates) * len(dropout_rates)  * len(activations) 
print(f'Running {total_runs} experiments')
exit()

current_run = 1
for ws in window_sizes:
	for feature in eeg_features:
		for model in models:
			for hc in hidden_channels:
				for nts in number_train_samples:
					for bs in batch_sizes:
						for lr in learning_rates:
							for dr in dropout_rates:
								for act in activations:

									bashCommand = f"python3 main.py -ws {ws} -ef {feature} -m {model} -hc {hc} -nts {nts} -bs {bs} -lr {lr} -act {act} -dr {dr} -wtr -esp 30 -trd cnn_test_1"
									print(f'Running ({current_run / total_runs}): {bashCommand}')
									process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
									output, error = process.communicate()
									print(output)