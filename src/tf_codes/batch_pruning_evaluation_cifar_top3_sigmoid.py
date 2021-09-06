#!/usr/bin/python

import tf_codes.robust_pruning_mlp_cifar_top3 as rp

SAMPLES = 1000

# Enable benchmarking mode will skip adversarial assessment to speed up the pruning process
benchmarking_mode = 0
activation = 'sigmoid'

hyperparameters = [0.05] # L1 norm in sigmoid mode is not important

# Omit baseline for the mass testing stage
rp.robust_pruning(mode='baseline', size=SAMPLES, benchmarking=benchmarking_mode, activation=activation, adversarial_epsilons=[0.01, 0.05])

for index, item in enumerate(hyperparameters):
    rp.robust_pruning(mode='stochastic', size=SAMPLES, 
        benchmarking=benchmarking_mode, activation=activation, alpha=item, adversarial_epsilons=[0.01, 0.05])

print('Task accomplished')