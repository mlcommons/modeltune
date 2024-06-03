import argparse

import pandas as pd
from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser(description="Test consensus")

    parser.add_argument("--datasets", type=str, nargs="+", help="Paths to the datasets")
    parser.add_argument("--output", type=str, help="Path to the output dataset")
    args = parser.parse_args()

    # Load datasets into pandas
    dataframes = [pd.read_csv(dataset) for dataset in args.datasets]

    # Drop unnecessary columns. We can add them back later if need be
    for i in range(len(dataframes)):
        dataframes[i] = dataframes[i].drop(
            columns=["original_category", "original_is_safe"]
        )

    # Join datasets on prompt/response pairs
    joined_dataframe = dataframes[0]
    for dataframe in dataframes[1:]:
        joined_dataframe = pd.merge(
            joined_dataframe, dataframe, on=["prompt", "response"], how="inner"
        )

    print("Shape:", joined_dataframe.shape)
    print("Info:")
    print(joined_dataframe.info())

    joined_dataframe.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
