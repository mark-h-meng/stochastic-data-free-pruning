#!/usr/bin/python

import subprocess

SAMPLES = 500

activation = 'sigmoid'

if activation == 'sigmoid':
    hyperparameters = ['0.0'] # L1 norm in sigmoid mode is not important
else:
    hyperparameters = ['0.5', '0.75']

output = subprocess.call(['python', 'robust_pruning_mlp_mnist.py',
                          '--mode', 'baseline',
                          '--activation', activation,
                          '--size', str(SAMPLES)], shell=True)

for index, item in enumerate(hyperparameters):
    output = subprocess.call(['python', 'robust_pruning_mlp_mnist.py',
                          '--mode', 'stochastic',
                          '--alpha', item,
                          '--activation', activation,
                          '--size', str(SAMPLES)], shell=True)

    output = subprocess.call(['python', 'robust_pruning_mlp_mnist.py',
                          '--mode', 'entropy',
                          '--alpha', item,
                          '--activation', activation,
                          '--size', str(SAMPLES)], shell=True)


print('Task accomplished')
