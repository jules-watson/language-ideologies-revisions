"""
Split the responses to prompts into configuration and stimuli.
"""
import pandas as pd
import csv

from constants import MODEL_NAME
from common import load_json
from split_algos import bleu_split

def load_processed(data_path):
    """
    Load the processed prompt responses from csv into a list of rows, each corresponding to a response
    """
    result = pd.read_csv(data_path)
    result["usage"] = [eval(item) for item in result["usage"]]
    return result


def split(config, processed, split_path, split_func_dict):
    """
    Split the processed sentences, given a dictionary of splitting functions corresponding to the task wording.
    """
    with open(split_path, 'w') as f:
        fieldnames = ["index"] + config["ind_var_cols"] + config["keep_cols"] + [
            "prompt", "finish_reason", "usage", "response", "id", "object", "created", "model"
        ] + ["revision", "justification"]
        csv_writer = csv.DictWriter(f, fieldnames=fieldnames)
        csv_writer.writeheader()

        for _, row in processed.iterrows():
            split_row = row.to_dict()

            # Split using corresponding splitting function; will return None, None if unsuccessful.
            split_func = split_func_dict[row["task_wording"]]
            split_row["revision"], split_row["justification"] = split_func(row["sentence"], row["response"])

            csv_writer.writerow(split_row)
        

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

    split_func_dict = {
        "simple": bleu_split,
    }

    split(config, processed, split_path, split_func_dict) 


if __name__ == "__main__":
    main("experiment/config.json")
