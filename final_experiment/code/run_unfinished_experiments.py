#!/usr/bin/env python

import os
import re
import torch
import time
from datetime import datetime
import subprocess
import csv
import numpy as np


with open('./unfinished.txt') as f:
	commands = f.readlines()
	# Number of experiments run in total
	total_runs = len(commands)
	print(f'Running {total_runs} experiments')
	# GPU setup
	max_n_running = (os.getenv('MAX_N_RUNNING') and int(os.getenv('MAX_N_RUNNING'))) or 1
	n_gpus, use_gpu_n = torch.cuda.device_count(), 0
	running_procs = []
	current_exp_n = 1
	for bashCommand in commands:
		# Run training job
		my_env = os.environ.copy()
		my_env['CUDA_VISIBLE_DEVICES'] = str(use_gpu_n)
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
			use_gpu_n = use_gpu_n if use_gpu_n < n_gpus else 0
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



