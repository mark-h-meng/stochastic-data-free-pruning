#!/usr/bin/python

import tf_codes.robust_pruning_mlp_mnist as rp

SAMPLES = 1000

# Enable benchmarking mode will skip adversarial assessment to speed up the pruning process
benchmarking_mode = 0
activation = 'relu'

hyperparameters = [0.75]

# Omit baseline for the mass testing stage
rp.robust_pruning(mode='baseline', size=SAMPLES, benchmarking=benchmarking_mode, activation=activation, batch_size_per_shot=2, pooling_multiplier=2, adversarial_epsilons=[0.01])

for index, item in enumerate(hyperparameters):
    rp.robust_pruning(mode='stochastic', size=SAMPLES, benchmarking=benchmarking_mode, activation=activation, alpha=item, batch_size_per_shot=2, pooling_multiplier=2, adversarial_epsilons=[0.01, 0.05, 0.1])

print('Task accomplished')
