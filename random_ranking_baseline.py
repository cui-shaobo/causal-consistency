"""
@Project  : fine-grained-defeasibility-in-causality
@File     : random_ranking_baseline.py
@Author   : Shaobo Cui
@Date     : 13.07.2024 16:06
"""
import csv
import random

if __name__ == '__main__':
    # Define the header and number of rows
    header = ["ID", "g-SD2", "g-SD1", "g-OD", "g-WD1", "g-WD2", "g-WS2", "g-WS1", "g-OS", "g-SS1", "g-SS2"]
    num_rows = 4000000

    # Open the CSV file to write
    with open('./results/results_random_baseline/random_ranking.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)

        # Loop to generate rows
        for i in range(num_rows):
            shuffled_list = list(range(10))
            random.shuffle(shuffled_list)
            writer.writerow([i] + shuffled_list)

    print("CSV file created successfully.")

