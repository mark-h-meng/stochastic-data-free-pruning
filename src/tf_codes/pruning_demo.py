#!/usr/bin/python3

import subprocess

# Set pruning_only to 1 to skip robustness evaluation (save time)
pruning_only = 1

# Specify the number of samples to be used in robustness evaluation (ignore this in pruning only mode)
SAMPLES = 1000

# Perform pruning of the cridit card fraud model (training goes first if saved model is not found)
output = subprocess.call(['python', 'robust_pruning_mlp_kaggle.py',
                          '--mode', 'stochastic',
                          '--alpha', '0.75',
                          '--benchmarking', str(pruning_only),
                          '--size', str(SAMPLES)], shell=True)

# Perform pruning of the chest x-ray pneumonia diagnosis model (training goes first if saved model is not found)
output = subprocess.call(['python', 'robust_pruning_mlp_chest_xray.py',
                          '--mode', 'stochastic',
                          '--alpha', '0.75',
                          '--benchmarking', str(pruning_only),
                          '--size', str(SAMPLES)], shell=True)

activation=['relu', 'sigmoid']
for index, item in enumerate(activation):
    alpha = '0.75'
    if activation=='sigmoid':
        alpha = '0.05'

    # Perform pruning of the MNIST model (training goes first if saved model is not found)
    output = subprocess.call(['python', 'robust_pruning_mlp_mnist.py',
                          '--mode', 'stochastic',
                          '--alpha', alpha,
                          '--activation', item,
                          '--benchmarking', str(pruning_only),
                          '--size', str(SAMPLES)], shell=True)

    # Perform pruning of the CIFAR-10 model (training goes first if saved model is not found)
    output = subprocess.call(['python', 'robust_pruning_mlp_cifar.py',
                          '--mode', 'stochastic',
                          '--alpha', alpha,
                          '--activation', item,
                          '--benchmarking', str(pruning_only),
                          '--size', str(SAMPLES)], shell=True)

print('Task accomplished')
