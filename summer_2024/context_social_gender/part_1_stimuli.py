"""
Generate stimuli similar to those used in psycholinguistic experiments for prompting LLMs.

This file is adapted from step_1_stimuli.py,
which can be found at: https://github.com/juliawatson/language-ideologies-revisions/blob/061012d9cd2d37e788734a709bbe9a8543d678d5/summer_2024/step_1_stimuli.py

Author: Xi (Joy) Wang
Date: Jan 2025
"""

import collections
import pandas as pd

import sys
import os

# Get the parent directory path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add parent directory to sys.path
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


from common import load_json
from constants import EXPERIMENT_PATH


def add_prompt(data, role_noun, role_noun_set, sentence, sentence_format, task_wording):
    """
    Add one row to the stimuli.csv file.
    """
    for task_wording_label, task_wording in task_wording.items():
        data["role_noun"].append(role_noun)
        data["role_noun_set"].append(role_noun_set)
        data["sentence"].append(sentence)
        data["sentence_format"].append(sentence_format)
        data["task_wording"].append(task_wording_label)
        final_sentence = f"{sentence} {task_wording}"
        data["prompt_text"].append(final_sentence)


def main_role_nouns(config, output_dir):
    """
    Create the stimuli file for role nouns.
    """
    sentence_data = pd.read_csv(config["sentence_data"])

    data = collections.defaultdict(list)

    neutral_role_noun_rows = sentence_data[sentence_data['role_noun_gender'].str.lower() == 'neutral']

    for _, row in neutral_role_noun_rows.iterrows():
        add_prompt(
                data=data,
                role_noun=row['role_noun'],
                role_noun_set=row['role_noun_set'],
                sentence=row['sentence'],
                sentence_format=row['sentence_format'],
                task_wording=config["task_wording"])
        

    df = pd.DataFrame(data)
    df.index.name = "index"
    output_path = f"{output_dir}/stimuli.csv"
    df.to_csv(output_path)


def main_mar7_full_analysis(config, output_dir):
    
    # load role noun sets
    role_nouns_df = pd.read_csv(config["role_nouns"])
    role_nouns_df["index"] = role_nouns_df["neutral"]
    role_nouns_df = role_nouns_df.set_index("index")

    # load sentence data
    about_me_data = pd.read_csv(config["sentence_data"], index_col=0)

    data = collections.defaultdict(list)
    role_noun_genders = ["neutral", "feminine", "masculine"]

    for _, row in about_me_data.iterrows():
        sentence = row['sentence_processed']
        role_noun_original = row["role_noun"]
        role_noun_neutral = row["role_noun_set"]
        role_noun_set = [
            role_nouns_df.loc[role_noun_neutral][curr_gender]
            for curr_gender in role_noun_genders
        ]

        # make sentence template
        assert sentence.count(role_noun_original) == 1
        sentence_format = sentence.replace(role_noun_original, '[ROLE NOUN]')
        new_sentence = sentence_format.replace('[ROLE NOUN]', role_noun_neutral)

        # This adds a prompt for the task wording
        add_prompt(
            data=data,
            role_noun=role_noun_neutral,
            role_noun_set=role_noun_set,
            sentence=new_sentence,
            sentence_format=sentence_format,
            task_wording=config["task_wording"])

    df = pd.DataFrame(data)
    df.index.name = "index"
    output_path = f"{output_dir}/stimuli.csv"
    df.to_csv(output_path)


def main(config_path):
    # Load the config
    config = load_json(config_path)
    config_dir = "/".join(config_path.split("/")[:-1])
    
    # Run the correct main function for the domain
    if config["domain"] == "role_nouns":
        # This is the default for now - it is for piloting and
        # downsamples sentences
        main_role_nouns(config, config_dir)
    elif config["domain"] == "mar7_full_analysis":
        main_mar7_full_analysis(config, config_dir)
    else:
        raise ValueError(f"Domain type not supported: {config['domain']}")
    

if __name__ == "__main__":
    main(f"{EXPERIMENT_PATH}/config.json")