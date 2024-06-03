"""
Goal it to take a teacher model dataset and clean it up for use in the consensus

1. Make sure categories are valid
    - Clean up categories column to be valid lists
    - Drop any non-categories listed
"""

import argparse
import pandas as pd
from constants import MLC_TAXONOMY_0_5_ALIASES

def main():
    parser = argparse.ArgumentParser(description="Clean a teacher model dataset for use in consensus")

    parser.add_argument('--dataset', type=str, help='Path to the dataset', required=True)
    parser.add_argument('--output', type=str, help='Path to the output dataset', required=True)
    args = parser.parse_args()

    # Load dataset
    df = pd.read_csv(args.dataset)

    # Iterate through categories column and clean them up
    # Find each safety category column for each teacher. Indicated by TEACHERMODELNAME_safety_categories
    # For each safety category column, go through each row, parse category str into a list, check that each item is in the set of valid category strings, throw out the ones that aren't
    for column in df.columns:
        if column.endswith("_safety_categories"):
            df[column] = df[column].apply(lambda x: [item.strip("[]' ") for item in x.strip("[]").split(",")])
            valid_categories = MLC_TAXONOMY_0_5_ALIASES  # Replace with your valid category strings
            df[column] = df[column].apply(lambda x: [item for item in x if item in valid_categories])
            # Getting to here implies that all items were in fact valid lists I believe
        
    # TODO, there may have been hallucinated categories that were dropped, leading to unsafe sample, but no safety categories. These need to be flagged

    # output the cleaned dataset
    df.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
