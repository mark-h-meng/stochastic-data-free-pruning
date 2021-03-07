#!/bin/bash
export QT_LOGGING_RULES='*=false'

python3 robust_pruning_mlp_kaggle.py --mode 'baseline' --benchmarking 1 --size 1000

declare -a params=('0.75' '0.25')

for alpha in ${params[@]}; do
  python3 robust_pruning_mlp_kaggle.py --mode 'stochastic' --alpha ${alpha} --benchmarking 1 --size 1000
done
