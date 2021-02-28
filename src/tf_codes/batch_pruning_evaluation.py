#!/usr/bin/python

import subprocess

SAMPLES = 500

pruning_only = 0
activation = 'sigmoid'

if activation == 'sigmoid':
    hyperparameters = ['0.0', '0.25'] # L1 norm in sigmoid mode is not important
else:
    hyperparameters = ['0.5', '0.75']

output = subprocess.call(['python', 'robust_pruning_mlp_mnist.py',
                          '--mode', 'baseline',
                          '--activation', activation,
                          '--benchmarking', str(pruning_only),
                          '--size', str(SAMPLES)], shell=True)

for index, item in enumerate(hyperparameters):
    output = subprocess.call(['python', 'robust_pruning_mlp_mnist.py',
                          '--mode', 'stochastic',
                          '--alpha', item,
                          '--activation', activation,
                          '--benchmarking', str(pruning_only),
                          '--size', str(SAMPLES)], shell=True)

print('Task accomplished')
