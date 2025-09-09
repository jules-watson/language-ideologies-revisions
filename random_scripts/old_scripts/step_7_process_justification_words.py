"""
Compare the justification word frequencies for different prompt types.

Author: Raymond Liu
Date: Aug 2024
"""

import pandas as pd

from common import load_json, load_csv, gender_list_dict
from constants import EXPERIMENT_PATH, MODEL_NAMES

prompt_genders = ['non_binary', 'man', 'woman', 'experiment_3_way']
role_noun_genders = ['neutral', 'masculine', 'feminine', 'all']

def process_data(words_list, output_path):
    """
    Create a CSV file combining the justification word frequencies.
    """
    result_dfs = {rn_gender: pd.DataFrame() for rn_gender in role_noun_genders}

    for prompt_gender in prompt_genders:
        for role_gender in role_noun_genders:
            file_path = f'piloting_{prompt_gender}/just_word_freqs_{role_gender}.csv'
            df = pd.read_csv(file_path)
            df = df[['word', 'average_frequency']]
            df = df.set_index('word')  # Set 'word' as index for easy merging
            df = df.reindex(words_list, fill_value=0.0)
            df = df.rename(columns={'average_frequency': f'{prompt_gender}_freq'})  # Rename the frequency column to prompt gender
            
            if result_dfs[role_gender].empty:
                result_dfs[role_gender] = df
            else:
                result_dfs[role_gender] = result_dfs[role_gender].join(df, how='outer')
    
    for role_gender in role_noun_genders:
        result_dfs[role_gender] = result_dfs[role_gender].fillna(0)
    
    for role_gender in role_noun_genders:
        output_file = f"{output_path}/{role_gender}_frequencies.csv"
        result_dfs[role_gender].to_csv(output_file)



def main(config):
    dirname = "/".join(config.split("/")[:-1])
    config_path = f"{dirname}/config.json"

    # Custom created words list
    words_list = ['professional', 'clarity', 'inclusive', 'tone', 'flow', 'concise', 'formal', 'engaging', 'clearer', 'sense',
              'specific', 'better', 'convey',
              'identity', 'readability', 'gender']

    output_path = f'justification_word_freq_results'
    
    process_data(words_list, output_path)

    
if __name__ == "__main__":
    main(f'{EXPERIMENT_PATH}/config.json')