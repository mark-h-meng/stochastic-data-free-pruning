#!/usr/bin/python

import tf_codes.robust_pruning_mlp_mnist as rp_mnist
import tf_codes.robust_pruning_mlp_cifar as rp_cifar
import tf_codes.robust_pruning_mlp_kaggle as rp_kaggle
import tf_codes.robust_pruning_mlp_chest_xray as rp_chest

SAMPLES = 1000

# Enable benchmarking mode will skip adversarial assessment to speed up the pruning process
benchmarking_mode = 1

hyperparameter_relu = 0.75
hyperparameter_sigmoid = 0.05

iteration_num = 1

while iteration_num > 0:
    rp_kaggle.robust_pruning(mode='stochastic', size=SAMPLES, benchmarking=benchmarking_mode, alpha=0.75)
    rp_chest.robust_pruning(mode='stochastic', size=SAMPLES, batch_size_per_shot=2, benchmarking=benchmarking_mode, alpha=0.75)
    rp_mnist.robust_pruning(mode='stochastic', size=SAMPLES, benchmarking=benchmarking_mode, activation='relu', alpha=0.75, batch_size_per_shot=2, pooling_multiplier=2)
    rp_mnist.robust_pruning(mode='stochastic', size=SAMPLES, benchmarking=benchmarking_mode, activation='sigmoid', alpha=0.05, batch_size_per_shot=2, pooling_multiplier=2)
    rp_cifar.robust_pruning(mode='stochastic', size=SAMPLES, benchmarking=benchmarking_mode, activation='relu', alpha=0.75, batch_size_per_shot=2, pooling_multiplier=2)
    rp_cifar.robust_pruning(mode='stochastic', size=SAMPLES, benchmarking=benchmarking_mode, activation='sigmoid', alpha=0.05, batch_size_per_shot=2, pooling_multiplier=2)
    iteration_num = iteration_num - 1

print('Task accomplished')
