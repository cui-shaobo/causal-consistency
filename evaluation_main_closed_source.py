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

    ideal_sequence = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    defeater_group = [0, 1, 2, 3, 4]
    supporter_group = [5, 6, 7, 8, 9]

    with open(args.input_file, mode='r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        ktd_defeater_group, ktd_supporter_group, ktd_total, cgp, igc = [], [], [], [], []

        for row in tqdm(reader):
            predicted_sequence = [row['g-SD2'], row['g-SD1'], row['g-OD'], row['g-WD1'], row['g-WD2'], row['g-WS2'], row['g-WS1'],
                                  row['g-OS'], row['g-SS1'], row['g-SS2']]
            for i, ranking in enumerate(predicted_sequence): predicted_sequence[i] = int(ranking)
            ktd_d, ktd_s, ktd_t, cgp_seq, igc_seq = calculate_seq_metric(predicted_sequence, ideal_sequence,
                                                                         defeater_group, supporter_group)
            ktd_defeater_group.append(ktd_d)
            ktd_supporter_group.append(ktd_s)
            ktd_total.append(ktd_t)
            cgp.append(cgp_seq)
            igc.append(igc_seq)
        
    mean_ktd_defeater_group, std_ktd_defeater_group = statistics.mean(ktd_defeater_group), statistics.stdev(
        ktd_defeater_group)
    mean_ktd_supporter_group, std_ktd_supporter_group = statistics.mean(ktd_supporter_group), statistics.stdev(
        ktd_supporter_group)
    mean_ktd_total, std_ktd_total = statistics.mean(ktd_total), statistics.stdev(ktd_total)
    mean_cgp, std_cgp = statistics.mean(cgp), statistics.stdev(cgp)
    mean_igc, std_igc = statistics.mean(igc), statistics.stdev(igc)

    print('mean ktd_supporter_group: {} \t std ktd_supporter_group: {}'.format(mean_ktd_supporter_group,
                                                                                std_ktd_supporter_group))
    print('mean ktd_defeater_group: {} \t std ktd_defeater_group: {}'.format(mean_ktd_defeater_group,
                                                                                std_ktd_defeater_group))
    print('mean ktd_total: {} \t std ktd_total: {}'.format(mean_ktd_total, std_ktd_total))
    print('mean cgp: {} \t std cgp: {}'.format(mean_cgp, std_cgp))
    print('mean igc: {} \t std igc: {}'.format(mean_igc, std_igc))

    print(print_latex_construction(mean_ktd_supporter_group, std_ktd_supporter_group,
                                      mean_ktd_defeater_group, std_ktd_defeater_group,
                                      mean_ktd_total, std_ktd_total,
                                      mean_cgp, std_cgp,
                                      mean_igc, std_igc))

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-name", help="Model Name", type=str, default="gpt-4o")
    parser.add_argument("--input-file", help="Ranking output path", type=str)

    args = parser.parse_args()
    if args.input_file is None:
        args.input_file = f"results/ranking_results_all_models/ranking_{args.model_name}.csv"
    calculate_all_metric(args)


if __name__ == '__main__':
    main()