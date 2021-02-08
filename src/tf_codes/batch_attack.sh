#!/bin/bash
export QT_LOGGING_RULES='*=false'

rub_pres=true

if $rub_pres ; then
  printf "\n"
	echo "Robustness preservation mode is ON!"
	declare -a models=('models/mnist_mlp' 'models/mnist_mlp_pruned_0.05' 'models/mnist_mlp_pruned_0.05_RobPres'
	  'models/mnist_mlp_pruned_0.1' 'models/mnist_mlp_pruned_0.1_RobPres'
	  'models/mnist_mlp_pruned_0.15' 'models/mnist_mlp_pruned_0.15_RobPres'
	  'models/mnist_mlp_pruned_0.2' 'models/mnist_mlp_pruned_0.2_RobPres'
	  'models/mnist_mlp_pruned_0.25' 'models/mnist_mlp_pruned_0.25_RobPres'
	  'models/mnist_mlp_pruned_0.3' 'models/mnist_mlp_pruned_0.3_RobPres'
	  'models/mnist_mlp_pruned_0.35' 'models/mnist_mlp_pruned_0.35_RobPres'
	  'models/mnist_mlp_pruned_0.4' 'models/mnist_mlp_pruned_0.4_RobPres'
	  'models/mnist_mlp_pruned_0.45' 'models/mnist_mlp_pruned_0.45_RobPres'
	  'models/mnist_mlp_pruned_0.5' 'models/mnist_mlp_pruned_0.5_RobPres')
else

	declare -a models=('models/mnist_mlp' 'models/mnist_mlp_pruned_0.05' 'models/mnist_mlp_pruned_0.1'
	  'models/mnist_mlp_pruned_0.15' 'models/mnist_mlp_pruned_0.2' 'models/mnist_mlp_pruned_0.25'
	  'models/mnist_mlp_pruned_0.3' 'models/mnist_mlp_pruned_0.35' 'models/mnist_mlp_pruned_0.4'
	  'models/mnist_mlp_pruned_0.45' 'models/mnist_mlp_pruned_0.5')
fi

for model in ${models[@]}; do
  printf "\n"
	echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
	echo "Running on ${model}"
	echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
	python3 adversarial_mnist_fgsm_batch.py --model ${model} --clean_output_folder 0 --shuffle 0 --adjust_gradient 1 --batch 1000
done
