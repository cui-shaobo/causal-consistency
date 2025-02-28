import argparse
from tqdm import tqdm
import csv
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def change_label(list):
    for i in range(len(list)):
        if int(list[i]) == 0: list[i] = "SD2"
        elif int(list[i]) == 1: list[i] = "SD1"
        elif int(list[i]) == 2: list[i] = "OD"
        elif int(list[i]) == 3: list[i] = "WD1"
        elif int(list[i]) == 4: list[i] = "WD2"
        elif int(list[i]) == 5: list[i] = "WS2"
        elif int(list[i]) == 6: list[i] = "WS1"
        elif int(list[i]) == 7: list[i] = "OS"
        elif int(list[i]) == 8: list[i] = "SS1"
        else: list[i] = "SS2"
    return list

def generate_confusion_matrix(args):

    with open(args.input_file, mode='r', newline='', encoding='utf-8') as csvfile:
        
        reader = csv.DictReader(csvfile)
        actual_rank, perceived_rank = [], []

        for row in tqdm(reader):
            perceived_rank.extend([row['g-SD2'], row['g-SD1'], row['g-OD'], row['g-WD1'], row['g-WD2'], row['g-WS2'], row['g-WS1'], row['g-OS'], row['g-SS1'], row['g-SS2']])
            actual_rank.extend([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

        actual_rank = change_label(actual_rank)
        perceived_rank = change_label(perceived_rank)

        cm = confusion_matrix(actual_rank,perceived_rank,labels=['SD2', 'SD1', 'OD', 'WD1', 'WD2', 'WS2', 'WS1', 'OS', 'SS1', 'SS2'])
        
        sns.heatmap(cm,
                    annot=True,
                    fmt='g',
                    xticklabels=['SD2', 'SD1', 'OD', 'WD1', 'WD2', 'WS2', 'WS1', 'OS', 'SS1', 'SS2'],
                    yticklabels=['SD2', 'SD1', 'OD', 'WD1', 'WD2', 'WS2', 'WS1', 'OS', 'SS1', 'SS2'])
        plt.ylabel('Prediction', fontsize=13)
        plt.xlabel('Actual', fontsize=13)
        plt.title(f'Confusion Matrix for Strength - {args.model_name}', fontsize=15)
        plt.savefig(args.output_file)  # Save the figure as a PDF with a tight layout
        plt.show()

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--model_name", help="OpenAI Model Name", type=str, default="gpt-4o")
    parser.add_argument("--input_file", help="Fine-grained Dataset Path", type=str)
    parser.add_argument("--output_file", help="Confusion Matrix Path", type=str)

    args = parser.parse_args()    
    if args.input_file is None:
        args.input_file = f"results/15_ranking_{args.model_name}.csv"
    if args.output_file is None:
        args.output_file = f"results/15_cm_{args.model_name}.pdf"

    generate_confusion_matrix(args)
    

if __name__ == '__main__':
    main()