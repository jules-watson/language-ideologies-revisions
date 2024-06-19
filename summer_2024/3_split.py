"""
Split the responses to prompts into configuration and stimuli.

Author: Raymond Liu
Date: May 2024
"""

import pandas as pd
import csv

from constants import MODEL_NAME, EXPERIMENT_PATH
from common import load_json
from split_algos import meteor_similarity_split

# Current splitting function
split_func_dict = {
    "meteor_similarity": meteor_similarity_split
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
    

def main(config):
    """
    For each response: split the response into revision and justification.
    """
    print(f"Splitting model: {MODEL_NAME}")
    print(f"config: {config}")

    dirname = "/".join(config.split("/")[:-1])
    config_path = f"{dirname}/config.json"
    processed_path = f"{dirname}/{MODEL_NAME}/processed.csv"
    split_path = f"{dirname}/{MODEL_NAME}/split.csv"

    config = load_json(config_path)
    processed = load_processed(processed_path)

    split(config, processed, split_path) 

    # Created a split_condensed.csv file with only the original sentence, revision, and justification
    split_condensed_path = f"{dirname}/{MODEL_NAME}/split_condensed.csv"
    condense(split_path, split_condensed_path, ['index', 'task_wording', 'sentence', 'revision', 'justification'])


if __name__ == "__main__":
    main(f"{EXPERIMENT_PATH}/config.json")
