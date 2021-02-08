#!/usr/bin/python

import subprocess

SAMPLES = 1000

hyperparameters = ['0.0', '0.25', '0.75', '1']

output = subprocess.call(['python', 'robust_pruning_mlp_chest_xray.py',
                          '--mode', 'baseline',
                          '--size', str(SAMPLES)], shell=True)

for index, item in enumerate(hyperparameters):
    output = subprocess.call(['python', 'robust_pruning_mlp_chest_xray.py',
                          '--mode', 'entropy',
                          '--alpha', item,
                          '--size', str(SAMPLES)], shell=True)

    output = subprocess.call(['python', 'robust_pruning_mlp_chest_xray.py',
                          '--mode', 'stochastic',
                          '--alpha', item,
                          '--size', str(SAMPLES)], shell=True)

print('Task accomplished')
