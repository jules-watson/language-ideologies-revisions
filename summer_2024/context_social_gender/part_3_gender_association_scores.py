"""
Analyze the gender associations of stimuli sentences.

Author: Xi (Joy) Wang
Date: Feb 2025
"""

import os, sys

# Get the parent directory path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add parent directory to sys.path
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from common import load_csv
from constants import EXPERIMENT_PATH, MODEL_NAME


def main(config_path):
    dirname = "/".join(config_path.split("/")[:-1])
    
    logprobs_path = f"{dirname}/{MODEL_NAME}/logprobs.csv"
    logprobs_data = load_csv(logprobs_path)
    output_path=f"{dirname}/{MODEL_NAME}"
    
    print(f"Calculating feminine score for: {MODEL_NAME}")
    # feminine_score = woman / (woman + man)
    logprobs_data["feminine_score"] = logprobs_data["woman"] / (logprobs_data["woman"] + logprobs_data["man"])

    print(f"Calculating genderedness score for: {MODEL_NAME}") 
    # genderedness_score = (woman + man) / (woman + man + person)
    logprobs_data["genderedness_score"] = (logprobs_data["woman"] + logprobs_data["man"]) / (logprobs_data["woman"] + logprobs_data["man"] + logprobs_data["person"])
                            
    logprobs_data.to_csv(f'{output_path}/sentence_gender_association_scores.csv', index=False)

if __name__ == "__main__":
    main(f"{EXPERIMENT_PATH}/config.json")