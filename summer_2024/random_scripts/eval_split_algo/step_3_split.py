"""
Split the responses to prompts into configuration and stimuli.

Author: Raymond Liu
Date: May 2024
"""

import pandas as pd
import csv
import os
import sys

summer_2024_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(summer_2024_dir)

from constants import EXPERIMENT_PATH, MODEL_NAMES
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
    

def main(processed_path, split_path, config_path):
    """
    For each response: split the response into revision and justification.
    """
    config = load_json(config_path)
    processed = load_processed(processed_path)

    split(config, processed, split_path) 

    # Created a split_condensed.csv file with only the original sentence, revision, and justification
    split_condensed_path = split_path.replace(".csv", "_condensed.csv")
    condense(split_path, split_condensed_path, ['index', 'sentence', 'revision', 'justification', 'model'])


if __name__ == "__main__":
    # processed_path = "train.csv"
    # split_path = "train_split.csv"
    # config_path = f"../../{EXPERIMENT_PATH}/config.json"
    # main(processed_path, split_path, config_path)

    processed_path = "test.csv"
    split_path = "test_split.csv"
    config_path = f"../../{EXPERIMENT_PATH}/config.json"
    main(processed_path, split_path, config_path)
