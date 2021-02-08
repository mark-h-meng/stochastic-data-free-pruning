#!/bin/bash
export QT_LOGGING_RULES='*=false'

declare -a modes=('baseline' 'greedy' 'entropy')

for mode in ${modes[@]}; do
  python3 robust_pruning_mlp_mnist.py --mode ${mode} --size 1000
done
