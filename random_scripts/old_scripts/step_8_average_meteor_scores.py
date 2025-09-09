"""
Calculate the average METEOR scores between original and revised sentences for different prompts types.

Author: Raymond Liu
Date: Aug 2024
"""


import pandas as pd

from common import load_json, load_csv, gender_list_dict
from constants import EXPERIMENT_PATH, MODEL_NAMES
from split_algos import calculate_meteor

prompt_genders = ['non_binary', 'man', 'woman', 'experiment_3_way']


def calc_avg_meteor_score(split_df, split_func_args):
    """
    Calculate the average METEOR score for a specific prompt split dataset
    """
    split_df['meteor_score'] = split_df.apply(lambda row: calculate_meteor(row['sentence'], row['revision'], split_func_args=split_func_args), axis=1)
    return split_df['meteor_score'].mean()


def main(config):
    dirname = "/".join(config.split("/")[:-1])
    config_path = f"{dirname}/config.json"
    config = load_json(config_path)

    for pg in prompt_genders:
        for model in MODEL_NAMES:
            split_df = pd.read_csv(f'piloting_{pg}/{model}/split.csv')
            avg_meteor_score = calc_avg_meteor_score(split_df, config['split_func_args'])
            print(f'For prompt gender {pg} and model {model}: average METEOR score is {avg_meteor_score}')

    
if __name__ == "__main__":
    main(f'{EXPERIMENT_PATH}/config.json')