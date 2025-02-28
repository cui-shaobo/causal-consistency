"""
@Project  : fine-grained-defeasibility-in-causality
@File     : confusion.py
@Author   : Shaobo Cui
@Date     : 24.05.2024 12:03
"""

from scipy.stats import kendalltau
import numpy as np
import argparse
from tqdm import tqdm
import csv
import statistics

def calculate_kendall_tau_distance(sequence, ideal_sequence):
    """ Calculate the Kendall tau distance which measures the number of discordant pairs. """
    mapping = {'SD2':0, 'SD1':1, 'OD':2, 'WD1':3, 'WD2':4, 'WS2':5, 'WS1':6, 'OS':7, 'SS1':8, 'SS2':9}
    if isinstance(sequence[0], str):
        sequence_int = [mapping[i] for i in sequence]
        ideal_sequence_int = [mapping[j] for j in ideal_sequence]
    else:
        sequence_int = sequence
        ideal_sequence_int = ideal_sequence
    tau, _ = kendalltau(sequence_int, ideal_sequence_int)
    scale_tau = (tau + 1.0) / 2.0
    # print(tau)
    return tau


def calculate_cross_group_position(sequence, G1, G2):
    """Calculate and normalize the Group Positional Bias."""
    violations = 0
    max_violations = len(G1) * len(G2)  # Every G2 element before every G1 element. The worst cases.
    # print(sequence, G1, G2)
    # Calculate actual violations
    for g2 in G2:
        g2_index = sequence.index(g2)
        for g1 in G1:
            g1_index = sequence.index(g1)
            if g2_index < g1_index:
                violations += 1

    # Normalize the GPB
    normalized_gpb = 1 - (violations / max_violations) if max_violations > 0 else 1

    return normalized_gpb


def calculate_intra_group_clustering(sequence, groups, weights=None):
    """
    Calculate clustering metrics for each group and aggregate them.

    Args:
    sequence (list): The complete sequence containing all elements.
    groups (list of lists): A list of groups, where each group is a list of elements.
    weights (list): Optional weights for each group, reflecting their relative importance.

    Returns:
    float: The aggregated clustering metric.
    """

    def calculate_group_metric(group):
        indices = sorted([sequence.index(x) for x in group])
        if len(indices) < 2:
            return 1  # Assume perfect clustering for single-element groups
        # Calculate the maximal distance between consecutive indices
        max_gap = max(indices[i + 1] - indices[i] for i in range(len(indices) - 1)) - 1
        # Calculate the density as the number of elements divided by the total span they occupy
        total_span = indices[-1] - indices[0] + 1  # +1 to account for the index zero start
        density = len(group) / total_span

        # Normalize the score to account for the max_gap
        if max_gap == 0:
            return density  # If there is no gap, density fully defines the score.
        else:
            # Adjustment factor considering the maximum gap
            return density * (1 - (max_gap / total_span))

    if weights is None:
        weights = [1] * len(groups)  # Equal weighting if none provided

    # Calculate metrics for all groups
    scores = [calculate_group_metric(group) for group in groups]

    # Aggregate metrics
    weighted_scores = [score * weight for score, weight in zip(scores, weights)]
    aggregated_metric = sum(weighted_scores) / sum(weights) if sum(weights) != 0 else 0

    return aggregated_metric

def calculate_seq_metric(predicted_sequence, ideal_sequence, defeater_group, supporter_group):
    # print(predicted_sequence)
    # print(ideal_sequence)
    # print(defeater_group)
    # print(supporter_group)
    split_index = len(defeater_group)
    ktd_defeater_group = calculate_kendall_tau_distance([x for x in predicted_sequence if x in defeater_group], defeater_group)
    ktd_supporter_group = calculate_kendall_tau_distance([x for x in predicted_sequence if x in supporter_group], supporter_group)
    ktd_total = calculate_kendall_tau_distance(predicted_sequence, ideal_sequence)
    cgp = calculate_cross_group_position(predicted_sequence, G1=defeater_group, G2=supporter_group)
    igc = calculate_intra_group_clustering(predicted_sequence, [defeater_group, supporter_group])
    # print('ktd_defeater_group: {}\nkdt_supporter_group: {}\nktd_total: {}\ncgp: {}\nigc: {}'.format(ktd_defeater_group, ktd_supporter_group, ktd_total, cgp, igc))

    return ktd_defeater_group, ktd_supporter_group, ktd_total, cgp, igc 

