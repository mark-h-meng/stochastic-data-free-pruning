#!/usr/bin/python3

import subprocess

SAMPLES = 1000

pruning_only = 0
hyperparameters = ['0.0','0.5','1']

output = subprocess.call(['python', 'robust_pruning_mlp_kaggle.py',
                          '--mode', 'baseline',
                          '--benchmarking', str(pruning_only),
                          '--size', str(SAMPLES)], shell=True)

for index, item in enumerate(hyperparameters):
    output = subprocess.call(['python', 'robust_pruning_mlp_kaggle.py',
                          '--mode', 'stochastic',
                          '--alpha', item,
                          '--benchmarking', str(pruning_only),
                          '--size', str(SAMPLES)], shell=True)

print('Task accomplished')
