import math
import progressbar
import numpy as np
import utility.interval_arithmetic as ia
import utility.simulated_propagation as simprop
import utility.utils as utils


def build_impact_dictionary(model, layer_loc, cumulative_impact_intervals, neuron_manipulated, l1_only=True):
    # Load the parameters and configuration of the input model
    (w, g) = utils.load_param_and_config(model)

    # score_dict is a dictionary contains all single units (in form of <node_a, -1>) and all unit pairs (in form of
    #    <node_a, node_b>, s.t. node_a > node_b)
    score_dict = {}
    curr_size = len(w[layer_loc][0][0])
    maxval_progress_bar = curr_size*(curr_size+1)/2

    if neuron_manipulated is None:
        neuron_manipulated = []

    bar = progressbar.ProgressBar(maxval=maxval_progress_bar,
                                  widgets=[progressbar.Bar('=', 'CALCULATING SCORES [', ']'), ' ', progressbar.Percentage()])
    bar.start()
    count = 0
    for node_a in range(0, curr_size):

        if node_a in neuron_manipulated:
            count += curr_size
            if count > maxval_progress_bar:
                count = maxval_progress_bar
            bar.update(count)
            continue

        # There is some issues that pruning single unit incurs smaller propagated impact but not good for robustness preservation
        #for node_b in range(-1, node_a):
        for node_b in range(0, node_a):

            count += 1
            if count > maxval_progress_bar:
                count = maxval_progress_bar
            bar.update(count)
            if node_b in neuron_manipulated:
                continue

            # Below is the hill climbing algorithm to update the top_candidate by dividing the original saliency by
            #   the l1-norm of the budget preservation list (the higher the better)
            pruning_impact_as_interval = simprop.calculate_impact_of_pruning_next_layer(model, big_map,
                                                                                        [(node_a, node_b)], 1)

            # Check is cumulative_impact_interval is none or not, not none means there is already some cumulative impact
            #   caused by previous pruning actions
            if cumulative_impact_intervals is not None:
                pruning_impact_as_interval = ia.interval_list_add(pruning_impact_as_interval,
                                                                  cumulative_impact_intervals)

            big_L = utils.l1_norm_of_intervals(pruning_impact_as_interval)
            if l1_only:
                score_dict[(node_a, node_b)] = big_L
            else:
                big_ENT = utils.interval_based_entropy(pruning_impact_as_interval, similarity_criteria=0.9)
                # Avoid entropy equals to zero
                score_dict[(node_a, node_b)] = 1 / (1 + math.exp(-1 * big_ENT)) * big_L

    bar.finish()

    # Sort the dictionary by-value before returining it to the invoking function
    score_dict = dict(sorted(score_dict.items(), key=lambda item: item[1]))

    return score_dict

