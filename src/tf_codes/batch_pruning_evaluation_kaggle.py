#!/usr/bin/python3

import subprocess

SAMPLES = 1000

hyperparameters = ['0.0','0.5','1']

output = subprocess.call(['python', 'robust_pruning_mlp_kaggle.py',
                          '--mode', 'baseline',
                          '--size', str(SAMPLES)], shell=True)

for index, item in enumerate(hyperparameters):
    output = subprocess.call(['python', 'robust_pruning_mlp_kaggle.py',
                          '--mode', 'entropy',
                          '--alpha', item,
                          '--size', str(SAMPLES)], shell=True)

    output = subprocess.call(['python', 'robust_pruning_mlp_kaggle.py',
                          '--mode', 'stochastic',
                          '--alpha', item,
                          '--size', str(SAMPLES)], shell=True)

print('Task accomplished')
