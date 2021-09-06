#!/usr/bin/python

import tf_codes.robust_pruning_mlp_chest_xray as rp

SAMPLES = 1000

# Enable benchmarking mode will skip adversarial assessment to speed up the pruning process
benchmarking_mode = 0

# Hyperparameters provide optional values for the "alpha". 
hyperparameters = [0.75]

# Omit baseline for the mass testing stage
rp.robust_pruning(mode='baseline', size=SAMPLES, batch_size_per_shot=8, benchmarking=benchmarking_mode)

for index, item in enumerate(hyperparameters):
    rp.robust_pruning(mode='stochastic', size=SAMPLES, batch_size_per_shot=8, benchmarking=benchmarking_mode, alpha=item)

print('Task accomplished')

