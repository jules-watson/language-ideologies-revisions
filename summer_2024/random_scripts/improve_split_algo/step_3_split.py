"""
Split the responses to prompts into configuration and stimuli.

Author: Raymond Liu
Date: May 2024
"""

import pandas as pd
import csv
import sys
import os

summer_2024_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(summer_2024_dir)

from constants import MODEL_NAME, EXPERIMENT_PATH
from common import load_json
from split_algos import meteor_similarity_split, meteor_heuristic_split

# Current splitting function
split_func_dict = {
    "meteor_similarity": meteor_similarity_split,
    "meteor_heuristic": meteor_heuristic_split
}


def load_processed(data_path):
    """
    Load the processed prompt responses from csv into a list of rows, each corresponding to a response
    """
    result = pd.read_csv(data_path)
    result["usage"] = [eval(item) for item in result["usage"]]
    return result


import pandas as pd

def performance_evaluation(split_path, gold_standard_path, index_col='index', revision_col='revision'):
    """
    Compare the revision column in the split file against a gold standard dataset to calculate and print the revision capture accuracy.
    """
    # Read the CSV files into dataframes
    split_df = pd.read_csv(split_path)
    gs_df = pd.read_csv(gold_standard_path)
    
    # Merge the dataframes on the index column
    merged_df = pd.merge(split_df, gs_df, on=index_col, suffixes=('_split', '_gold_standard'))
    
    # Compare the revision columns
    matches = merged_df[revision_col + '_split'] == merged_df[revision_col + '_gold_standard']
    
    # Calculate the accuracy
    accuracy = matches.sum() / len(matches)
    
    # Print the accuracy
    print(f'Revision capture accuracy: {accuracy:.2f}')


def split(config, processed, split_path):
    """
    Split the processed sentences, given a dictionary of splitting functions corresponding to the task wording.
    """
    split_func = split_func_dict[config["split_func"]]
    split_func_args = config["split_func_args"]

    with open(split_path, 'w') as f:
        fieldnames = ["index"] + config["ind_var_cols"] + config["keep_cols"] + [
            "prompt", "finish_reason", "usage", "response", "id", "object", "created", "model"
        ] + ["revision", "justification"]
        csv_writer = csv.DictWriter(f, fieldnames=fieldnames)
        csv_writer.writeheader()

        for _, row in processed.iterrows():
            split_row = row.to_dict()

            # Split using corresponding splitting function; will return None, None if unsuccessful.
            split_row["revision"], split_row["justification"] = split_func(row["sentence"], row["response"], split_func_args)
            csv_writer.writerow(split_row)
        

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

    dirname = "../../" + "/".join(config.split("/")[:-1])
    print(f"dirname: {dirname}")
    config_path = f"{dirname}/config.json"
    print(f"config_path: {config_path}")
    processed_path = f"{dirname}/{MODEL_NAME}/processed.csv"
    print(f"processed_path: {processed_path}")
    split_path = f"./test_split_llama.csv"
    print(f"split_path: {split_path}")

    config = load_json(config_path)
    processed = load_processed(processed_path)

    split(config, processed, split_path) 

    # Created a split_condensed.csv file with only the original sentence, revision, and justification
    # split_condensed_path = f"./split_with_heuristic_condensed.csv"
    # condense(split_path, split_condensed_path, ['index', 'sentence', 'revision', 'justification'])

    # calculate the revision capture accuracy of the split
    gold_standard_path = f"{dirname}/{MODEL_NAME}/gold_standard_split_dataset.csv"
    performance_evaluation(split_path, gold_standard_path)


if __name__ == "__main__":
    main(f"{EXPERIMENT_PATH}/config.json")
