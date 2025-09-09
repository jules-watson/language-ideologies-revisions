"""
Analyze the revisions for certain characteristics.

Author: Raymond Liu
Date: Aug 2024
"""

import pandas as pd
import json
import ast
import re

from common import load_json, load_csv, gender_list_dict
from constants import EXPERIMENT_PATH, MODEL_NAMES


def analyze_revisions(split_data, role_nouns_path, output_path):
    """
    Add columns to each revision row:
    - whether or not original variant was removed
    - the gender of the variant within the revised sentence
    """
    # Fill NaN values in 'revision' with empty strings
    split_data['revision'] = split_data['revision'].fillna('')

    if role_nouns_path.endswith(".json"):
        with open(role_nouns_path, 'r') as f:
            role_nouns = json.load(f)
            gender_list = gender_list_dict[len(role_nouns[0])]
    else:  # it's a csv
        assert role_nouns_path.endswith(".csv")
        role_nouns_df = pd.read_csv(role_nouns_path)
        role_nouns_df["index"] = role_nouns_df["neutral"]
        role_nouns_df = role_nouns_df.set_index("index")

    split_data['variant_removed'] = split_data.apply(
        lambda row: re.search(
                r'\b(' + row['role_noun'] +  r')\b', row['revision'], re.IGNORECASE
            ) is None, axis=1)


    # Function to check if the sentence contains any role noun other than the first column role noun
    def check_variant_addition(row):
        role_nouns = ast.literal_eval(row['role_noun_set'])
        primary_role = row['role_noun']

        for i, role in enumerate(role_nouns):
            role_regex = re.compile(r'\b(' + role +  r')\b', re.IGNORECASE)
            if role != primary_role and role_regex.search(row['revision']):
                
                if role_nouns_path.endswith(".json"):
                    return gender_list[i]
                else:
                    assert role_nouns_path.endswith(".csv")
                    neutral_role = role_nouns[0]
                    row = role_nouns_df.loc[neutral_role]
                    if role == neutral_role:
                        return "neutral"
                    if role == row["feminine"]:
                        return "feminine"
                    if role == row["masculine"]:
                        return "masculine"

        return 'None'

    split_data['variant_added'] = split_data.apply(check_variant_addition, axis=1)

    split_data.to_csv(f'{output_path}/revision.csv',index=True)

    return split_data


def compute_stats(rev_data, output_path):
    """
    Compute statistics for each role noun gender:
     - how often the role noun is removed
     - how often a different role noun variant is substituted, for each potential variant gender
    """
    gender_list = gender_list_dict[3]

    proportions_data = []

    # perform proportion calculations for each prompt wording
    for prompt_wording, prompt_df in rev_data.groupby("task_wording"):
        # calculate proportions for each gender variant
        for gender in gender_list:
            filtered_data = prompt_df[prompt_df['role_noun_gender'] == gender]
            proportions_row = {
                'prompt_wording': prompt_wording,
                'starting_variant': gender,
                'removed_rate': ((filtered_data['variant_removed']) & (filtered_data['variant_added'] == 'None')).mean()
            }

            for subbed_gender in gender_list:
                proportions_row[f'{subbed_gender}_added'] = (filtered_data['variant_added'] == subbed_gender).mean()
            
            proportions_data.append(proportions_row)

    proportions_df = pd.DataFrame(proportions_data)

    proportions_df.to_csv(f'{output_path}/revision_stats.csv', index=False)


def main(config_path):

    print(f"config: {config_path}")
    dirname = "/".join(config_path.split("/")[:-1])
    config = load_json(config_path)

    for model_name in MODEL_NAMES:

        print(f"Calculating revision statistics for: {model_name}")
        
        split_path = f"{dirname}/{model_name}/split.csv"
        split_data = load_csv(split_path)
        role_nouns_path = config['role_nouns']
        output_path=f"{dirname}/{model_name}"

        revision_data = analyze_revisions(split_data, role_nouns_path, output_path)
        compute_stats(revision_data, output_path)


if __name__ == "__main__":
    main(f"{EXPERIMENT_PATH}/config.json")