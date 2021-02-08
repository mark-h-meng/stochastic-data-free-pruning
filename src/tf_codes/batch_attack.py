#!/usr/bin/python

import subprocess

ROBUST_PRES=False
SAMPLES = 20

models = ['models/mnist_mlp',
          'models/mnist_mlp_pruned_0.05',
          'models/mnist_mlp_pruned_0.15',
          'models/mnist_mlp_pruned_0.25',
          'models/mnist_mlp_pruned_0.35',
          'models/mnist_mlp_pruned_0.45',
          'models/mnist_mlp_pruned_0.1',
          'models/mnist_mlp_pruned_0.2',
          'models/mnist_mlp_pruned_0.3',
          'models/mnist_mlp_pruned_0.4',
          'models/mnist_mlp_pruned_0.5']

for index, model in enumerate(models):
    if index==0 or not ROBUST_PRES:
        output = subprocess.call(['python', 'adversarial_mnist_fgsm_batch.py',
                          '--model', model,
                          '--clean_output_folder', '0',
                          '--shuffle', '0',
                          '--adjust_gradient', '1',
                          '--batch', str(SAMPLES)], shell=True)
    else:
        output = subprocess.call(['python', 'adversarial_mnist_fgsm_batch.py',
                          '--model', model,
                          '--clean_output_folder', '0',
                          '--adjust_gradient', '1',
                          '--shuffle', '0',
                          '--batch', str(SAMPLES)], shell=True)

        output = subprocess.call(['python', 'adversarial_mnist_fgsm_batch.py',
                          '--model', model+"_RobPres",
                          '--clean_output_folder', '0',
                          '--adjust_gradient', '1',
                          '--shuffle', '0',
                          '--batch', str(SAMPLES)], shell=True)
print('Task accomplished')