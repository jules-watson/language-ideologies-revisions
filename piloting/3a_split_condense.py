"""
Condense the split.csv file to create a new file only showing the post-split revisions and justifications.

Author: Raymond Liu
Date: May 2024
"""

import pandas as pd

from constants import MODEL_NAME


def condense(split_path, split_condensed_path, cols):
    split_df = pd.read_csv(split_path)
    
    # Select only the specified columns
    split_condensed_df = split_df[cols]
    
    # Write the filtered dataframe to a new CSV file
    split_condensed_df.to_csv(split_condensed_path, index=False)



def main(config):
    """
    For each response: split the response into revision and justification.
    """
    print(f"Splitting model: {MODEL_NAME}")
    print(f"config: {config}")

    dirname = "/".join(config.split("/")[:-1])
    config_path = f"{dirname}/config.json"
    split_path = f"{dirname}/{MODEL_NAME}/split.csv"
    split_condensed_path = f"{dirname}/{MODEL_NAME}/split_condensed.csv"

    condense(split_path, split_condensed_path, ['task_wording', 'revision', 'justification'])

if __name__ == "__main__":
    main("experiment/config.json")