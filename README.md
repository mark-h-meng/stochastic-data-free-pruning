Our python code can run pruning progressively and evaluate robustness automatically.

You may run our demonstrative source code to perform training and pruning of all six models in one run, which is exactly same with our experiment 1:

* src/tf_codes/pruning_demo.py

You may also execute one of the following source code to run the pruning with FGSM robustness evaluation after each iteration (it may run for a few hours), depending on which dataset & model you would like to try:

> Please be noted in each python/bash file below, there is an argument called "benchmarking". You can disable it by setting to zero to speedup the pruning process. Enabling it will lead a FGSM attack followed by robustness evaluation after each pruning epoch, which could be very slow.


*In case you are running a Windows machine (Tested on Windows 10):*

* src/tf_codes/batch_pruning_evaluation.py (MNIST, MLP)
* src/tf_codes/batch_pruning_evaluation_chest.py (Chest X-ray, CNN)
* src/tf_codes/batch_pruning_evaluation_kaggle.py (Credit Card Fraud, MLP)
* src/tf_codes/batch_pruning_evaluation_cifar.py (CIFAR-10, CNN)

*If you are running a Linux/Mac machine, you may use alternative below (Tested on Ubuntu 20.04 LTS):*

* src/tf_codes/batch_pruning_evaluation_mnist.sh (MNIST, MLP)
* src/tf_codes/batch_pruning_evaluation_chest.sh (Chest X-ray, CNN)
* src/tf_codes/batch_pruning_evaluation_kaggle.sh (Credit Card Fraud, MLP)
* src/tf_codes/batch_pruning_evaluation_cifar.sh (CIFAR-10, CNN)

You can also donwload the source code as a ZIP package through the sharing link below:

* https://seto.teracloud.jp/share/1191706bb0b64346 
(You don't have to log in to download, just click "continue without logging in" at the bottom)
