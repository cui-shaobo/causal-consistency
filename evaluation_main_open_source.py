"""
@Project  : fine-grained-defeasibility-in-causality
@File     : evaluation_main.py
@Author   : Shaobo Cui
@Date     : 04.07.2024 11:43
"""
import argparse
import csv
import statistics
from tqdm import tqdm
import pandas as pd
from metrics.confusion import calculate_seq_metric


def print_latex_construction(mean_ktd_supporter_group, std_ktd_supporter_group,
                             mean_ktd_defeater_group, std_ktd_defeater_group,
                             mean_ktd_total, std_ktd_total,
                             mean_cgp, std_cgp,
                             mean_igc, std_igc):
    latex_format = (
        "{:.3f} \\stdvalue{{{:.3f}}} & "
        "{:.3f} \\stdvalue{{{:.3f}}} & "
        "{:.3f} \\stdvalue{{{:.3f}}} & "
        "{:.3f} \\stdvalue{{{:.3f}}} & "
        "{:.3f} \\stdvalue{{{:.3f}}} \\\\"
    ).format(
        mean_ktd_supporter_group, std_ktd_supporter_group,
        mean_ktd_defeater_group, std_ktd_defeater_group,
        mean_ktd_total, std_ktd_total,
        mean_cgp, std_cgp,
        mean_igc, std_igc
    )
    return latex_format


def calculate_all_metric(args):
    # ideal_sequence = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    # defeater_group = ['0', '1', '2', '3', '4']
    # supporter_group = ['5', '6', '7', '8', '9']
    ideal_sequence = ['SD2', 'SD1', 'OD', 'WD1', 'WD2', 'WS2', 'WS1', 'OS', 'SS1', 'SS2']
    defeater_group = ['SD2', 'SD1', 'OD', 'WD1', 'WD2']
    supporter_group = ['WS2', 'WS1', 'OS', 'SS1', 'SS2']
    mapping = {0: 'SD2', 1: 'SD1', 2: 'OD', 3: 'WD1', 4: 'WD2', 5: 'WS2', 6: 'WS1', 7: 'OS', 8: 'SS1', 9: 'SS2'}

    reader = pd.read_csv(args.input_file)

    ktd_defeater_group, ktd_supporter_group, ktd_total, cgp, igc = [], [], [], [], []

    reader = reader.applymap(lambda x: mapping[x] if x in mapping else x)

    for i, row in tqdm(reader.iterrows()):
        # print('index: {}, row: {}'.format(i, row))
        predicted_sequence = [row['g-SD2'], row['g-SD1'], row['g-OD'], row['g-WD1'], row['g-WD2'], row['g-WS2'],
                              row['g-WS1'],
                              row['g-OS'], row['g-SS1'], row['g-SS2']]
        if set(predicted_sequence) == set(['WS1', 'SD1', 'SS1', 'WS2', 'SD2', 'WD1', 'WD2', 'SS2', 'OD', 'OS']):
            ktd_d, ktd_s, ktd_t, cgp_seq, igc_seq = calculate_seq_metric(predicted_sequence, ideal_sequence,
                                                                         defeater_group, supporter_group)

            ktd_defeater_group.append(ktd_d)
            ktd_supporter_group.append(ktd_s)
            ktd_total.append(ktd_t)
            cgp.append(cgp_seq)
            igc.append(igc_seq)
        else:
            pass

    mean_ktd_defeater_group, std_ktd_defeater_group = statistics.mean(ktd_defeater_group), statistics.stdev(
        ktd_defeater_group)
    mean_ktd_supporter_group, std_ktd_supporter_group = statistics.mean(ktd_supporter_group), statistics.stdev(
        ktd_supporter_group)
    mean_ktd_total, std_ktd_total = statistics.mean(ktd_total), statistics.stdev(ktd_total)
    mean_cgp, std_cgp = statistics.mean(cgp), statistics.stdev(cgp)
    mean_igc, std_igc = statistics.mean(igc), statistics.stdev(igc)

    # print('mean ktd_supporter_group: {} \t std ktd_supporter_group: {}'.format(mean_ktd_supporter_group,
    #                                                                            std_ktd_supporter_group))
    # print('mean ktd_defeater_group: {} \t std ktd_defeater_group: {}'.format(mean_ktd_defeater_group,
    #                                                                          std_ktd_defeater_group))
    # print('mean ktd_total: {} \t std ktd_total: {}'.format(mean_ktd_total, std_ktd_total))
    # print('mean cgp: {} \t std cgp: {}'.format(mean_cgp, std_cgp))
    # print('mean igc: {} \t std igc: {}'.format(mean_igc, std_igc))
    latex_format = print_latex_construction(mean_ktd_supporter_group, std_ktd_supporter_group,
                                            mean_ktd_defeater_group, std_ktd_defeater_group,
                                            mean_ktd_total, std_ktd_total,
                                            mean_cgp, std_cgp,
                                            mean_igc, std_igc)
    print(latex_format)


def calculate_all_metric_function(input_file):
    # ideal_sequence = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    # defeater_group = ['0', '1', '2', '3', '4']
    # supporter_group = ['5', '6', '7', '8', '9']
    ideal_sequence = ['SD2', 'SD1', 'OD', 'WD1', 'WD2', 'WS2', 'WS1', 'OS', 'SS1', 'SS2']
    defeater_group = ['SD2', 'SD1', 'OD', 'WD1', 'WD2']
    supporter_group = ['WS2', 'WS1', 'OS', 'SS1', 'SS2']
    mapping = {0: 'SD2', 1: 'SD1', 2: 'OD', 3: 'WD1', 4: 'WD2', 5: 'WS2', 6: 'WS1', 7: 'OS', 8: 'SS1', 9: 'SS2'}

    reader = pd.read_csv(input_file)

    ktd_defeater_group, ktd_supporter_group, ktd_total, cgp, igc = [], [], [], [], []

    reader = reader.applymap(lambda x: mapping[x] if x in mapping else x)

    for i, row in tqdm(reader.iterrows()):
        # print('index: {}, row: {}'.format(i, row))
        predicted_sequence = [row['g-SD2'], row['g-SD1'], row['g-OD'], row['g-WD1'], row['g-WD2'], row['g-WS2'],
                              row['g-WS1'],
                              row['g-OS'], row['g-SS1'], row['g-SS2']]
        if set(predicted_sequence) == set(['WS1', 'SD1', 'SS1', 'WS2', 'SD2', 'WD1', 'WD2', 'SS2', 'OD', 'OS']):
            ktd_d, ktd_s, ktd_t, cgp_seq, igc_seq = calculate_seq_metric(predicted_sequence, ideal_sequence,
                                                                         defeater_group, supporter_group)

            ktd_defeater_group.append(ktd_d)
            ktd_supporter_group.append(ktd_s)
            ktd_total.append(ktd_t)
            cgp.append(cgp_seq)
            igc.append(igc_seq)
        else:
            pass

    mean_ktd_defeater_group, std_ktd_defeater_group = statistics.mean(ktd_defeater_group), statistics.stdev(
        ktd_defeater_group)
    mean_ktd_supporter_group, std_ktd_supporter_group = statistics.mean(ktd_supporter_group), statistics.stdev(
        ktd_supporter_group)
    mean_ktd_total, std_ktd_total = statistics.mean(ktd_total), statistics.stdev(ktd_total)
    mean_cgp, std_cgp = statistics.mean(cgp), statistics.stdev(cgp)
    mean_igc, std_igc = statistics.mean(igc), statistics.stdev(igc)

    # print('mean ktd_supporter_group: {} \t std ktd_supporter_group: {}'.format(mean_ktd_supporter_group,
    #                                                                            std_ktd_supporter_group))
    # print('mean ktd_defeater_group: {} \t std ktd_defeater_group: {}'.format(mean_ktd_defeater_group,
    #                                                                          std_ktd_defeater_group))
    # print('mean ktd_total: {} \t std ktd_total: {}'.format(mean_ktd_total, std_ktd_total))
    # print('mean cgp: {} \t std cgp: {}'.format(mean_cgp, std_cgp))
    # print('mean igc: {} \t std igc: {}'.format(mean_igc, std_igc))
    latex_format = print_latex_construction(mean_ktd_supporter_group, std_ktd_supporter_group,
                                            mean_ktd_defeater_group, std_ktd_defeater_group,
                                            mean_ktd_total, std_ktd_total,
                                            mean_cgp, std_cgp,
                                            mean_igc, std_igc)
    return latex_format

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-name", help="OpenAI Model Name", type=str, default="llama-7B")
    parser.add_argument("--input-file", help="Ranking output path", type=str, default="results/condiprob_results_of_opensource_models/ranking_gemma-2B_Therefore.csv")

    args = parser.parse_args()
    calculate_all_metric(args)


if __name__ == '__main__':
    main()
