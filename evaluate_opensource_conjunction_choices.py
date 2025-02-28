"""
@Project  : fine-grained-defeasibility-in-causality
@File     : evaluate_opensource_conjunction_choices.py
@Author   : Shaobo Cui
@Date     : 27.07.2024 11:37
"""

import subprocess

# Define your model name
from evaluation_main_open_source import calculate_all_metric_function

modelName = "llama3-70B"

# List of conjunction words
conjunctionwords = ['so', 'because', 'since', 'as', 'therefore', 'thus', 'hence']
output_texts = []

# Iterate over each conjunction word and run the script
for word in conjunctionwords:
    input_file = f"results/condiprob_results_of_opensource_models/ranking_{modelName}_{word}.csv"
    latex_output_format = calculate_all_metric_function(input_file)
    # Print the output
    output_texts.append("``" + word + "''" + " & " + latex_output_format)

for output_text in output_texts:
    print(output_text)