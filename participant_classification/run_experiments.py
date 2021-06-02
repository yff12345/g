#!/usr/bin/env python

import subprocess

# fields = "model - eeg_feature - remove_global_connections - remove_baseline_signal_noise_removal - number_test_targets - number_validation_targets - batch_size - hidden_channels - learning_rate - mean_test_loss - test_acc - test_f1 - test_prec - test_reca - test_roc - pytorch_total_params"

models = ['GraphConv','GatedGraphConv','GCN']
eeg_features = ['wav','psd']
# remove_global_connections = [False,True]
# remove_baseline_signal_noise_removal = [False,True]
number_test_targets = [10, 30, 34]
batch_sizes = [4, 16]
hidden_channels = [16, 64 ,128]
learning_rates = [0.0001, 0.0045, 0.0090]
dropout_rates = [0, 0.2, 0.4]


for model in models:
	for eeg_feature in eeg_features:
		for number_test_target in number_test_targets:
			for hidden_channel in hidden_channels:
				for lr in learning_rates:
					for dr in dropout_rates:
						for bs in batch_sizes:
							bashCommand = f"python3 main.py -m {model} -ef {eeg_feature} -ntt {number_test_target} -bs {bs} -hc {hidden_channel} -lr {lr} -dr {dr} -st -wtr -esp 30"
							print(f'Running: {bashCommand}')
							process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
							output, error = process.communicate()
							print(output)