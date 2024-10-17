"""
Analyze the revisions for certain characteristics.

Author: Raymond Liu
Date: Aug 2024
"""

import pandas as pd
import json
import ast

from common import load_json, load_csv, gender_list_dict
from constants import EXPERIMENT_PATH, MODEL_NAME


def analyze_revisions(split_data, role_nouns_path, output_path):
    """
    Add columns to each revision row:
    - whether or not original variant was removed
    - the gender of the variant within the revised sentence
    """
    with open(role_nouns_path, 'r') as f:
        role_nouns = json.load(f)
        gender_list = gender_list_dict[len(role_nouns[0])]
    
    split_data['variant_removed'] = split_data.apply(lambda row: row['role_noun'] not in row['revision'], axis=1)

    # Function to check if the sentence contains any role noun other than the first column role noun
    def check_variant_addition(row):
        role_nouns = ast.literal_eval(row['role_noun_set'])
        primary_role = row['role_noun']

        for i, role in enumerate(role_nouns):
            if role != primary_role and role in row['revision']:
                return gender_list[i]
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
        print(f'Analyzing prompt wording {prompt_wording}')
        # calculate proportions for each gender variant
        for gender in gender_list:
            print(f'\tAnalyzing gender {gender}')
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
    print(proportions_df)

    proportions_df.to_csv(f'{output_path}/revision_stats.csv', index=False)


def main(config):
    print(f"Calculating revision statistics for: {MODEL_NAME}")

    dirname = "/".join(config.split("/")[:-1])
    config_path = f"{dirname}/config.json"
    split_path = f"{dirname}/{MODEL_NAME}/split.csv"

    config = load_json(config_path)

    split_data = load_csv(split_path)
    role_nouns_path = config['role_nouns']
    output_path=f"{dirname}/{MODEL_NAME}"

    revision_data = analyze_revisions(split_data, role_nouns_path, output_path)
    compute_stats(revision_data, output_path)


if __name__ == "__main__":
    main(f"{EXPERIMENT_PATH}/config.json")