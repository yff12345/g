#!/usr/bin/env python

import subprocess

# fields = "model - eeg_feature - remove_global_connections - remove_baseline_signal_noise_removal - number_test_targets - number_validation_targets - batch_size - hidden_channels - learning_rate - mean_test_loss - test_acc - test_f1 - test_prec - test_reca - test_roc - pytorch_total_params"

models = ['GraphConv','GatedGNN','NOGNN']
eeg_features = ['wav','psd']
remove_global_connections = False
remove_baseline_signal_noise_removal = False
number_test_targets = [10, 20, 30, 34]
number_validation_targets = 5
batch_size = 4
hidden_channels = [16,32,128]
learning_rates = [1e-2, 1e-3, 1e-4]


for model in models:
	for eeg_feature in eeg_features:
		for number_test_target in number_test_targets:
			for hidden_channel in hidden_channels:
				for lr in learning_rates:
					bashCommand = f"python3 main.py -m {model} -ef {eeg_feature} -ntt {number_test_target} -bs {batch_size} -hc {hidden_channel} -lr {lr} -st -wtr -esp 30"
					print(f'Running: {bashCommand}')
					process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
					output, error = process.communicate()
					print(output)