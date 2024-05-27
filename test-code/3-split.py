"""
Split the responses to prompts into configuration and stimuli.
"""
import pandas as pd
import csv
import re

from constants import MODEL_NAME
from common import load_config

def load_processed(data_path):
    """
    Load the processed prompt responses from csv into a list of rows, each corresponding to a response
    """
    result = pd.read_csv(data_path)
    result["form_set"] = [eval(item) for item in result["form_set"]]
    result["usage"] = [eval(item) for item in result["usage"]]
    return result


def newline_split(response):
    """
    Strategy: Split on the first occurence of \n. Before that is revision, after is justification
    """
    pos = response.find('\n')

    if pos != -1:
        return response[:pos].strip(), response[pos+2:].strip()
    else:
        return None, None
    

def label_split(response):
    """
    Strategy: Revision is string in between the substrings 'Sentence' and 'Explanation', 
    including puncutation immediately around those substrings.
    Justification is after 'Explanation'.
    """    
    rev_match = re.search("([\*]*)Sentence([: \*]*)", response)
    just_match = re.search("([\*]*)Explanation([: \*]*)", response)
    if rev_match and just_match:
        return response[rev_match.end():just_match.start()].strip(), response[just_match.end():].strip()
    else:
        return None, None


def punc_split(response):
    """
    Strategy: revision sentence is in square brackets. Justification is after the ending square bracket
    """
    rev_start = response.find("[")
    rev_end = response.find("]")
    if rev_start != -1 and rev_end != -1:
        return response[rev_start+1:rev_end].strip(), response[rev_end+1:].strip()
    else:
        return None, None



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
            split_row["revision"], split_row["justification"] = split_func(row["response"])

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

    config = load_config(config_path)
    processed = load_processed(processed_path)

    split_func_dict = {
        "direct": newline_split,
        "label": label_split,
        "punc": punc_split,
    }

    split(config, processed, split_path, split_func_dict) 


if __name__ == "__main__":
    main("test-experiment/config.json")
