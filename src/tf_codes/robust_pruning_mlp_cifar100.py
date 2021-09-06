#!/usr/bin/python3

# Import publicly published & installed packages
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import adversarial_mnist_fgsm_batch as adversarial
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import FormatStrFormatter
import argparse

from numpy.random import seed
import os, time, csv, sys, shutil, math, time

# Import own classes
import tf_codes.training_from_data as training_from_data
import tf_codes.pruning as pruning
import tf_codes.utility.utils as utils
import tf_codes.utility.bcolors as bcolors
import tf_codes.utility.interval_arithmetic as ia
import tf_codes.utility.simulated_propagation as simprop

# Specify a random seed
seed(42)
tf.random.set_seed(42)

# Specify the mode of pruning
BASELINE_MODE = False
BENCHMARKING_MODE = False

# Recursive mode
# PS: Baseline should is written in non-recursive mode
RECURSIVE_PRUNING = False

# E.g. EPOCHS_PER_CHECKPOINT = 5 means we save the pruned model as a checkpoint after each five
#    epochs and at the end of pruning
EPOCHS_PER_CHECKPOINT = 30


def robust_pruning(mode='entropy', size=100, alpha=0.5, activation='relu', benchmarking=0,
    target_pruning_percentage=0.8, adversarial_epsilons=[0.01, 0.05], batch_size_per_shot=2, pooling_multiplier=4):

    curr_mode = mode
    BATCH_SIZE_PER_EVALUATION = size
    hyper_parameter_alpha = alpha
    activation = activation
    BENCHMARKING_MODE = benchmarking > 0

    hyper_parameter_beta = 1-hyper_parameter_alpha
    hyperparameters = (hyper_parameter_alpha, hyper_parameter_beta)


    # ReLU is the default option of activation (will be updated later, just set Ture here)
    use_relu = True

    # E.g. TARGET_PRUNING_PERCENTAGE = 0.3 means 30% of hidden units are expected to be pruned off
    TARGET_PRUNING_PERCENTAGE = target_pruning_percentage

    # E.g. TARGET_ADV_EPSILON = 0.3 means the maximum perturbation epsilon that we expect our pruned
    #    model to preserve is 0.3 (only applicable to normalized input)
    #TARGET_ADV_EPSILONS = [0.01, 0.025, 0.05, 0.1, 0.25]
    TARGET_ADV_EPSILONS = adversarial_epsilons

    # E.g. PRUNING_SIZE (PER_EPOCH_PER_LAYER) = 3 means we only choose 2 out of hidden units as the set
    #    of candidates at each layer during each round of pruning
    BATCH_SIZE_PER_PRUNING = batch_size_per_shot
    POOLING_MULTIPLIER = pooling_multiplier


    if curr_mode == 'baseline':
        BASELINE_MODE = True
        print(bcolors.OKCYAN)
        print('>' * 50)
        print(">> CASE: CIFAR-100 ("+activation+")")
        print(">> BASELINE MODE: pruning by saliency only")
        print(">> ACTIVATION: " + str(activation))
        print(">> EVALUATION BATCH SIZE: " + str(BATCH_SIZE_PER_EVALUATION))
        if BENCHMARKING_MODE:
            print(">> BENCHMARKING MODE ENABLED")
        print('>' * 50)
        print(bcolors.ENDC)
    elif curr_mode == 'entropy':
        print(bcolors.OKCYAN)
        print('>' * 50)
        print(">> CASE: CIFAR-100 ("+activation+")")
        BASELINE_MODE = False
        print(">> ENTROPY MODE: pruning by entropy")
        print(">> ACTIVATION: " + str(activation))
        print(">> EVALUATION BATCH SIZE: " + str(BATCH_SIZE_PER_EVALUATION))
        print(">> HYPER-PARAMETER (ALPHA): " + str(hyper_parameter_alpha))
        if BENCHMARKING_MODE:
            print(">> BENCHMARKING MODE ENABLED")
        print('>' * 50)
        print(bcolors.ENDC)
    elif curr_mode == 'stochastic':
        BASELINE_MODE = False
        print(bcolors.OKCYAN)
        print('>' * 50)
        print(">> CASE: CIFAR-100 ("+activation+")")
        print(">> STOCHASTIC MODE: pruning by entropy with simulated annealing")
        print(">> EVALUATION BATCH SIZE: " + str(BATCH_SIZE_PER_EVALUATION))
        print(">> HYPER-PARAMETER (ALPHA): " + str(hyper_parameter_alpha))
        if BENCHMARKING_MODE:
            print(">> BENCHMARKING MODE ENABLED")
        print('>' * 50)
        print(bcolors.ENDC)
    else:
        print(bcolors.FAIL)
        print('>' * 50)
        print(">> UNRECOGNIZED MODE: please check your input and try again")
        print(">> Accepted input: baseline, entropy or stochastic")
        print('>' * 50)
        print(bcolors.ENDC)
        return

    utils.create_dir_if_not_exist("tf_codes/logs/")
    utils.create_dir_if_not_exist("tf_codes/save_figs/")

    # Define a list to record each pruning decision
    tape_of_moves = []

    # Define a hash map to store definition intervals for all FC neurons
    big_map = {}

    # Define a list to record benchmark & evaluation per pruning epoch (begins with original model)
    score_board = []
    accuracy_board = []

    epoch_couter = 0
    num_units_pruned = 0

    original_model_path = 'tf_codes/models/cifar_100_mlp_9_layer'
    pruned_model_path = 'tf_codes/models/cifar_100_mlp_pruned_9_layer'
    
    if activation == 'sigmoid':
        original_model_path += '_sigmoid'
        pruned_model_path += '_sigmoid'
        use_relu = False

    ################################################################
    # Prepare dataset and pre-trained model                        #
    ################################################################
    # The CIFAR-100 dataset contains 60,000 32x32 color images in 10 classes.
    # There are 50000 training images and 10000 test images.
    data_pre_path = 'tf_codes/input/kaggle-cifar-100/' 

    (train_images, train_labels), (test_images, test_labels), (valid_images, valid_labels), (subCategoryDict, superCategoryDict) = training_from_data.load_data_cifar100(data_pre_path)

    print("Training dataset size: ", train_images.shape, train_labels.shape)


    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    training_from_data.train_cifar_100_9_layer_cnn((train_images, train_labels),
                                                   (test_images, test_labels),
                                                   (valid_images, valid_labels),
                                                   original_model_path,
                                                   overwrite=False,
                                                   use_relu=use_relu,
                                                   optimizer_config=optimizer)
    num_units_first_mlp_layer = 256

    model = tf.keras.models.load_model(original_model_path)
    print(model.summary())

    #model.compile(optimizer=optimizer, loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])
    
    model.compile(optimizer=optimizer, 
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                    metrics=['sparse_top_k_categorical_accuracy'])
        
    loss, accuracy = model.evaluate(test_images, test_labels, verbose=2)

    # TEMP IMPLEMENTATION STARTS HERE
    big_map = simprop.get_definition_map(model, input_interval=(0,1))
    # TEMP IMPLEMENTATION ENDS HERE

    if BASELINE_MODE and not BENCHMARKING_MODE:
        robust_preservation = adversarial.robustness_evaluation_cifar_topK(model,
                                                                (test_images, test_labels),
                                                                TARGET_ADV_EPSILONS,
                                                                BATCH_SIZE_PER_EVALUATION, k=5)
        score_board.append(robust_preservation)
        accuracy_board.append((round(loss, 4), round(accuracy, 4)))

        tape_of_moves.append([])
        print(bcolors.OKGREEN, "[Original]", str(robust_preservation), bcolors.ENDC)

    ################################################################
    # Launch a pruning epoch                                       #
    ################################################################

    percentage_been_pruned = 0
    stop_condition = False
    neurons_manipulated =None
    target_scores = None
    pruned_pairs = None
    cumulative_impact_intervals = None
    saliency_matrix=None

    # Start elapsed time counting
    start_time = time.time()

    while(not stop_condition):

        # The list neurons_manipulated records all neurons have been involved in pruning as a pair, and
        #   passed to pruning function by-reference.
        if BASELINE_MODE:
            model, neurons_manipulated, pruned_pairs, saliency_matrix = pruning.pruning_baseline(model,
                                                        big_map,
                                                        prune_percentage=BATCH_SIZE_PER_PRUNING/num_units_first_mlp_layer,
                                                        neurons_manipulated=neurons_manipulated,
                                                        saliency_matrix=saliency_matrix,
                                                        recursive_pruning=False,
                                                        bias_aware=True)

            count_pairs_pruned_curr_epoch = 0
            if pruned_pairs is not None:
                for layer, pairs in enumerate(pruned_pairs):
                    if len(pairs) > 0:
                        print(" >> Pruning", pairs, "at layer", str(layer))
                        for pair in pairs:
                            count_pairs_pruned_curr_epoch += 1

        elif curr_mode == 'entropy':
            pruning_result = pruning.pruning_greedy(model,
                                                    big_map,
                                                    prune_percentage=BATCH_SIZE_PER_PRUNING/num_units_first_mlp_layer,
                                                    cumulative_impact_intervals=cumulative_impact_intervals,
                                                    pooling_multiplier=POOLING_MULTIPLIER,
                                                    neurons_manipulated=neurons_manipulated,
                                                    hyperparamters=hyperparameters,
                                                    recursive_pruning=True,
                                                    bias_aware=True)

            (model, neurons_manipulated, pruned_pairs, cumulative_impact_intervals, score_dicts) = pruning_result

            count_pairs_pruned_curr_epoch = 0
            if pruned_pairs is not None:
                for layer, pairs in enumerate(pruned_pairs):
                    if len(pairs) > 0:
                        print(" >> Pruning", pairs, "at layer", str(layer))
                        print(" >>   with assessment score ", end=' ')
                        for pair in pairs:
                            count_pairs_pruned_curr_epoch += 1
                            print(round(score_dicts[layer][pair], 3), end=' ')
                        print()
        # For the case curr_mode=='stochastic':
        else:
            pruning_result = pruning.pruning_stochastic(model,
                                                        big_map,
                                                        prune_percentage=BATCH_SIZE_PER_PRUNING / 128,
                                                        cumulative_impact_intervals=cumulative_impact_intervals,
                                                        neurons_manipulated=neurons_manipulated,
                                                        target_scores=target_scores,
                                                        hyperparamters=hyperparameters,
                                                        recursive_pruning=True)

            (model, neurons_manipulated, target_scores, pruned_pairs, cumulative_impact_intervals,
             score_dicts) = pruning_result

            count_pairs_pruned_curr_epoch = 0
            if pruned_pairs is not None:
                for layer, pairs in enumerate(pruned_pairs):
                    if len(pairs) > 0:
                        print(" >> Pruning", pairs, "at layer", str(layer))
                        print(" >>   with assessment score ", end=' ')
                        for pair in pairs:
                            count_pairs_pruned_curr_epoch += 1
                            print(round(score_dicts[layer][pair], 3), end=' ')
                        print()
                        print(" >> Updated target scores at this layer:", round(target_scores[layer], 3))

        epoch_couter += 1

        # Check if the list of pruned pair is empty or not - empty means no more pruning is feasible
        if count_pairs_pruned_curr_epoch == 0:
            stop_condition = True
            print(" >> No more hidden unit could be pruned, we stop at EPOCH", epoch_couter)
        else:
            if not BASELINE_MODE:
                print(" >> Cumulative impact as intervals after this epoch:")
                print(cumulative_impact_intervals)

            percentage_been_pruned += BATCH_SIZE_PER_PRUNING/num_units_first_mlp_layer
            print(" >> Pruning progress:", bcolors.BOLD, str(percentage_been_pruned * 100) + "%", bcolors.ENDC)
            num_units_pruned += count_pairs_pruned_curr_epoch
            print(" >> Total number of units pruned:", bcolors.BOLD, num_units_pruned, bcolors.ENDC)

            model.compile(optimizer=optimizer, loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])
            if not BENCHMARKING_MODE:
                robust_preservation = adversarial.robustness_evaluation_cifar_topK(model,
                                                                        (test_images, test_labels),
                                                                        TARGET_ADV_EPSILONS,
                                                                        BATCH_SIZE_PER_EVALUATION, k=5)

                loss, accuracy = model.evaluate(test_images, test_labels, verbose=2)

                # Update score_board and tape_of_moves
                score_board.append(robust_preservation)
                accuracy_board.append((round(loss, 4), round(accuracy, 4)))

                print(bcolors.OKGREEN + "[Epoch " + str(epoch_couter) + "]" + str(robust_preservation) + bcolors.ENDC)

            tape_of_moves.append(pruned_pairs)
            pruned_pairs = None

        # Check if have pruned enough number of hidden units
        if BASELINE_MODE and percentage_been_pruned >= 0.5:
            print(" >> Maximum pruning percentage has been reached")
            stop_condition = True
        elif not stop_condition and percentage_been_pruned >= TARGET_PRUNING_PERCENTAGE:
            print(" >> Target pruning percentage has been reached")
            stop_condition = True

        # Save the pruned model at each checkpoint or after the last pruning epoch
        if epoch_couter % EPOCHS_PER_CHECKPOINT == 0 or stop_condition:
            curr_pruned_model_path = pruned_model_path + "_ckpt_" + str(hyper_parameter_alpha) + "_" + str(math.ceil(epoch_couter/EPOCHS_PER_CHECKPOINT))

            if os.path.exists(curr_pruned_model_path):
               shutil.rmtree(curr_pruned_model_path)
               print("Removed existing pruned model ...")

            model.save(curr_pruned_model_path)
            print(" >>> Pruned model saved")

    # Stop elapsed time counting
    end_time = time.time()
    print("Elapsed time: ", round((end_time - start_time)/60.0, 3), "minutes /", int(end_time - start_time), "seconds")

    ################################################################
    # Save the tape of moves                                       #
    ################################################################
    
    # Obtain a timestamp
    local_time = time.localtime()
    timestamp = time.strftime('%b-%d-%H%M', local_time)

    tape_filename = "tf_codes/logs/cifar100-" + activation + "-" + timestamp + "-" + str(BATCH_SIZE_PER_EVALUATION)
    if BENCHMARKING_MODE:
        tape_filename = tape_filename+"-BENCHMARK"

    if BASELINE_MODE:
        tape_filename += "_tape_baseline.csv"
    else:
        tape_filename = tape_filename + "_tape_" + curr_mode + "_" + str(hyper_parameter_alpha) + ".csv"

    while os.path.exists(tape_filename):
        # os.remove(tape_filename)
        tape_filename = tape_filename[:-4] + "a.csv"

    with open(tape_filename, 'w+', newline='') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',')

        csv_line = [str(eps) for eps in TARGET_ADV_EPSILONS]
        csv_line.append('moves,loss,accuracy')
        csv_writer.writerow(csv_line)

        for index, item in enumerate(score_board):
            rob_pres_stat = [item[k] for k in TARGET_ADV_EPSILONS]
            rob_pres_stat.append(tape_of_moves[index])
            rob_pres_stat.append(accuracy_board[index])
            csv_writer.writerow(rob_pres_stat)

        if BENCHMARKING_MODE:
            csv_writer.writerow(["Elapsed time: ", round((end_time - start_time) / 60.0, 3), "minutes /", int(end_time - start_time), "seconds"])
