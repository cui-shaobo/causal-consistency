#!/bin/bash

# Model tag
model_tag="llama2-70B"

# List of conjunctions
conjunctions=('so' 'because' 'since' 'as' 'therefore' 'thus' 'hence')

# Loop through each conjunction
for conjunction in "${conjunctions[@]}"; do
    echo "Processing conjunction: $conjunction"
    # Run the specified line of code with the current conjunction
    python ranking_with_conjunctions.py --model_tag "$model_tag" --conjunction "$conjunction"
done


python openSourceGenerations/llama3/generate.py

