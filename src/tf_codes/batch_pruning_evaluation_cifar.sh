#!/bin/bash
export QT_LOGGING_RULES='*=false'

python3 robust_pruning_mlp_cifar.py --mode 'baseline' --activation 'sigmoid' --benchmarking 1 --size 1000

declare -a sigmoidparams=('0.05' '0.25')

for alpha in ${sigmoidparams[@]}; do
  python3 robust_pruning_mlp_cifar.py --mode 'stochastic' --alpha ${alpha} --activation 'sigmoid' --benchmarking 1 --size 500
done

declare -a reluparams=('0.5' '0.75')

for alpha in ${reluparams[@]}; do
  python3 robust_pruning_mlp_cifar.py --mode 'stochastic' --alpha ${alpha} --activation 'relu' --benchmarking 1 --size 500
done