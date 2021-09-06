Our python code can run pruning progressively and evaluate robustness automatically. The code is written and tested through Microsoft VS Code.

The execution environment of the experiments on paper is summarised as follows: 
* Tensorflow 2.3.0 (Anaconda, tensorflow-gpu version)
* Python 3.8
* Ubuntu 20.04 LTS



You may run our demonstrative source code to perform training and pruning of all six models in one run, which is exactly same with our experiment (the root path of this project is "src/"):

* src/tf_codes/quick_benchmark.py

You may also execute one of the following source code to run the pruning with FGSM robustness evaluation after each iteration (it may run for a few hours), depending on which dataset & model you would like to try:

> Please be noted in each python/bash file, there is an argument called "benchmarking". You can disable it by setting to 1 to speedup the pruning process. Disabling it (set to 0) will lead a FGSM attack followed by robustness evaluation after each pruning epoch, which could be slow.

You can also run a specific model pruning and robustness evalution by executing an arbitrary file listed below (Tested on Ubuntu 20.04 LTS):

* src/tf_codes/batch_pruning_evaluation_kaggle.sh (Credit Card Fraud, MLP, 2 labels)
* src/tf_codes/batch_pruning_evaluation_chest.sh (Chest X-ray, CNN, 2 labels)
* src/tf_codes/batch_pruning_evaluation_mnist_relu.py (MNIST-ReLU, MLP, 10 labels)
* src/tf_codes/batch_pruning_evaluation_mnist_sigmoid.py (MNIST-Sigmoid, MLP, 10 labels)
* src/tf_codes/batch_pruning_evaluation_cifar_relu.py (CIFAR-10-ReLU, CNN, 10 labels)
* src/tf_codes/batch_pruning_evaluation_cifar_sigmoid.py (CIFAR-10-Sigmoid, CNN, 10 labels)

In addition, we also evaluate the robustness in top-K mode (K=3 in our implementation), you may try to run codes below:

* src/tf_codes/batch_pruning_evaluation_cifar_top3_relu.py (CIFAR-10-ReLU, CNN, 10 labels)
* src/tf_codes/batch_pruning_evaluation_cifar_top3_sigmoid.py (CIFAR-10-Sigmoid, CNN, 10 labels)

All the models trained and pruned will be saved in tf_codes/model directory.

All the log files recording the pruning details at each step, accuracy and robustness assessment (if not in benchmarking mode) are saved with time stamp in filename in tf_codes/logs directory.

You can also donwload the source code as a ZIP package through the sharing link below (the zipped file contains offline datasets so that it will save your time at the first time execution, while training the original models):

* https://seto.teracloud.jp/share/1191706bb0b64346 
(You don't have to log in to download, just click "continue without logging in" at the bottom)
