#!/usr/bin/python

import subprocess

SAMPLES = 1000

pruning_only = 0
hyperparameters = ['0.1',  '0.75']

output = subprocess.call(['python', 'robust_pruning_mlp_chest_xray.py',
                          '--mode', 'baseline',
                          '--benchmarking', str(pruning_only),
                          '--size', str(SAMPLES)], shell=True)

for index, item in enumerate(hyperparameters):
    output = subprocess.call(['python', 'robust_pruning_mlp_chest_xray.py',
                          '--mode', 'stochastic',
                          '--alpha', item,
                          '--benchmarking', str(pruning_only),
                          '--size', str(SAMPLES)], shell=True)

print('Task accomplished')
