import argparse
import pandas as pd
from tqdm import tqdm

from conditionalProbModel.ConditionalProbGemma2B import ConditionalProbGemma2B
from conditionalProbModel.ConditionalProbGemma7B import ConditionalProbGemma7B
from conditionalProbModel.ConditionalProbLLaMASecond13B import ConditionalProbLLaMASecond13B
from conditionalProbModel.ConditionalProbLLaMASecond70B import ConditionalProbLLaMASecond70B
from conditionalProbModel.ConditionalProbLLaMASecond7B import ConditionalProbLLaMASecond7B
from conditionalProbModel.ConditionalProbLLaMAThird70B import ConditionalProbLLaMAThird70B
from conditionalProbModel.ConditionalProbMistral7B import ConditionalProbMistral7B
from conditionalProbModel.ConditionalProbPhiMedium import ConditionalProbPhiMedium
from conditionalProbModel.ConditionalProbPhiMini import ConditionalProbPhiMini
from conditionalProbModel.ConditionalProbPhiSmall import ConditionalProbPhiSmall
from conditionalProbModel.utils import construct_conjunction


def construct_inferencers(args):
    if args.model_tag == 'llama2-7B':
        inferencer = ConditionalProbLLaMASecond7B(args)
    elif args.model_tag == 'llama2-13B':
        inferencer = ConditionalProbLLaMASecond13B(args)
    elif args.model_tag == 'llama2-70B':
        inferencer = ConditionalProbLLaMASecond70B(args)
    elif args.model_tag == 'gemma-2B':
        inferencer = ConditionalProbGemma2B(args)
    elif args.model_tag == 'gemma-7B':
        inferencer = ConditionalProbGemma7B(args)
    elif args.model_tag == 'mistral-7B':
        inferencer = ConditionalProbMistral7B(args)
    elif args.model_tag == 'phi3-3.8B':
        inferencer = ConditionalProbPhiMini(args)
    elif args.model_tag == 'phi3-7B':
        inferencer = ConditionalProbPhiSmall(args)
    elif args.model_tag == 'phi3-14B':
        inferencer = ConditionalProbPhiMedium(args)
    elif args.model_tag == 'llama3-70B':
        inferencer = ConditionalProbLLaMAThird70B(args)

    return inferencer

def rank_values(row):
    # Sort values and get ranks
    sorted_indices = row.argsort()
    ranks = sorted_indices.argsort()
    return ranks

def value_to_ranking(condiprob_df):
    columns_to_rank = ["SD2", "SD1", "D", "WD1", "WD2", "WS2", "WS1", "S", "SS1", "SS2"]

    # DataFrame to hold the results
    results_df = pd.DataFrame()
    results_df["ID"] = condiprob_df["ID"]

    # Apply the ranking function to each row for the specified columns
    ranks = condiprob_df[columns_to_rank].apply(rank_values, axis=1)

    # Assign the ranks to the appropriate columns in the results DataFrame
    for col, rank_col in zip(columns_to_rank, ranks.columns):
        if col == 'D' or col == 'S':
            results_df[f"g-O{col}"] = ranks[rank_col]
        else:
            results_df[f"g-{col}"] = ranks[rank_col]

    return results_df

def main():
    parser = argparse.ArgumentParser(description="Calculate conditional probabilities for a specific model and conjunction word.")
    parser.add_argument("--cuda", action="store_false", default=True)
    parser.add_argument("--model_tag", default="gemma-2B", choices=["llama2-7B", "llama2-13B", "gemma-2B", "gemma-7B", "mistral-7B", "phi3-3.8B", "phi3-7B", "phi3-14B", "llama3-70B"], required=False, help="The model to use for calculation.")
    parser.add_argument("--conjunction", default='therefore', choices=['so', 'because', 'since', 'as', 'therefore', 'thus', 'hence'], required=False, help="The conjunction word to use.")
    # parser.add_argument("--csv_path", default='./results/generation_results_of_opensource_models/output-gemma-7b-it.csv', required=False, help="Path to the input CSV file.")
    args = parser.parse_args()

    condiprob_inferencer = construct_inferencers(args)
    input_csv_path = f'./results/generation_results_of_opensource_models/output-opensource-generation-{args.model_tag}.csv'
    # Read CSV file
    df = pd.read_csv(input_csv_path)

    intermediate_columns = ["SD2", "SD1", "D", "WD1", "WD2", "WS2", "WS1", "S", "SS1", "SS2"]

    # Prepare results DataFrame
    results_list = []

    # Process each row in the CSV
    for index, row in tqdm(df.iterrows()):
        result_row = {"ID": row['ID']}
        for intermediate in intermediate_columns:
            if pd.notna(row[intermediate]):
                conditional_input, conditional_output = construct_conjunction(cause_text=row['cause'], intermediate_text=row[intermediate], conjunction_word=args.conjunction, effect_text=row['long_term_effect'])
                # print(conditional_input, conditional_output)
                result = condiprob_inferencer.calculate_conditional_probability(conditional_input, conditional_output)
                result_row[intermediate] = result['conditional_prob'].item()
            else:
                result_row[intermediate] = None
        results_list.append(result_row)

    condiprob_results_df = pd.DataFrame(results_list)

    # Save results to new CSV file
    condiprob_output_csv_path = f"./results/condiprob_results_of_opensource_models/conditional_prob_{args.model_tag}_{args.conjunction}.csv"
    condiprob_results_df.to_csv(condiprob_output_csv_path, index=False)
    print(f"Results saved to {condiprob_output_csv_path}")

    ranking_results_df = value_to_ranking(condiprob_results_df)
    ranking_output_csv_path = f"./results/condiprob_results_of_opensource_models/ranking_{args.model_tag}_{args.conjunction}.csv"
    ranking_results_df.to_csv(ranking_output_csv_path, index=False)
    print(f"Results saved to {ranking_output_csv_path}")

if __name__ == "__main__":
    main()
