#!/usr/bin/env python

import os
import re
import torch
import time
from datetime import datetime
import subprocess
import csv
import numpy as np

# Fixed hyperparameters
lr = 5e-4
l2 = 0 
dr = 0.25

# Experiments
experiment_n = list(range(10))
features = ['raw']
window_sizes = [1, 2]
number_train_samples = [16, 32, 64, 128]
# These should use the same training data
model_names = ['GraphConv']
hidden_channels = [512, 1024]
models = list(filter(None, [f'{name}_{hc}' if not name == 'LR' else None if hc != 64 else 'LR' for name in model_names for hc in hidden_channels ]))
experiments = [(en,ef,ws,nts) for en in experiment_n for ef in features for ws in window_sizes for nts in number_train_samples]

# Number of experiments run in total
total_runs = len(experiments) * len(models)
print(f'Running {total_runs} experiments')

max_n_running = (os.getenv('MAX_N_RUNNING') and int(os.getenv('MAX_N_RUNNING'))) or 1
n_gpus, use_gpu_n = torch.cuda.device_count(), 1
running_procs = []
current_exp_n = 1

for i,(en,ef,ws,nts) in enumerate(experiments):
	# Get train data indices for this set of experiments
	samples_per_participant = int((60/ws)*40)
	# First 100 are validation indices, then train
	val_train_samples_idx = np.random.default_rng().choice(samples_per_participant, size=nts+100, replace=False).astype(str)
	
	# Pick nts samples randomly
	for model in models:
		# Get model name and hidden channels
		if model == 'LR':
			hc = 0
		else:
			model, hc = model.split('_')

		# Run training job
		my_env = os.environ.copy()
		my_env['CUDA_VISIBLE_DEVICES'] = str(use_gpu_n)
		bashCommand = f"python3 main.py -ws {ws} -ef {ef} -hc {hc} -nts {nts} -wd {l2} -dr {dr} -m {model} -lr {lr} -wtr -esp 30 -trd {model}_{hc}_{ef}_{nts}_{ws} -tmd {current_exp_n} -dsd -tsa {' '.join(val_train_samples_idx) }"
		process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE, env=my_env)
		print(f'Running ({current_exp_n} / {total_runs}) GPU({str(use_gpu_n)}) PID({process.pid}) ({datetime.now().strftime("%D %H:%M:%S")}): {bashCommand} \n')
		running_procs.append(process)
		current_exp_n += 1		

		while len(running_procs) >= max_n_running:
			# print('Max exp running...', end = "\r")
			still_running = []
			for proc in running_procs:
				res = proc.poll()
				if res is None:
					still_running.append(proc)
				else:
					print(f'-Process {proc.pid} finished running (OK)-')
			running_procs = still_running
			time.sleep(1)

		# Switch to next GPU. 
		if current_exp_n <= max_n_running:
			use_gpu_n += 1
			use_gpu_n = use_gpu_n if use_gpu_n < n_gpus else 1
		# Use the one with most ammount of free memory
		else:
			most_free_mem = 0 
			for gpu_n in range(n_gpus):
				bashCommand = f'nvidia-smi --query-gpu=memory.free --format=csv -i {gpu_n} | grep -Eo [0-9]+'
				process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
				output, error = process.communicate()
				if error == None:
					free_mem_mb = int(re.search("[0-9]+", str(output))[0])
					if free_mem_mb > most_free_mem:
						most_free_mem = free_mem_mb
						use_gpu_n = gpu_n
				else:
					print(error)
					exit()



		