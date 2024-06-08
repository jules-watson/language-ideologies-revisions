"""
Generate the stimuli file of prompts to be queried by LLMs.

This file is inspired from part_1_stimuli.py,
which can be found at: https://github.com/juliawatson/language-ideologies-2024/blob/main/fall_2023_main/part_1_stimuli.py

Author: Raymond Liu
Date: Jun 2024
"""

import collections
import pandas as pd

from common import load_json
from constants import AN_NOUNS


def add_prompts(data, 
                name, name_gender,
                role_noun, role_noun_gender, role_noun_label,
                sentence, task_wording):
    """
    Add one row to the stimuli.csv file.
    """
    for task_wording_label, task_wording in task_wording.items():
        data["name"].append(name)
        data["name_gender"].append(name_gender)

        data["role_noun"].append(role_noun)
        data["role_noun_gender"].append(role_noun_gender)
        data["role_noun_label"].append(role_noun_label)

        data["task_wording"].append(task_wording_label)
        
        final_sentence = f"{task_wording} {sentence}"
        data["sentence"].append(sentence)
        data["prompt_text"].append(final_sentence)



def main_role_nouns(config, output_dir):
    """
    Create the stimuli file for role nouns.
    """
    names = load_json(config["names"])
    role_nouns = load_json(config["role_nouns"]) 

    data = collections.defaultdict(list)

    # Iterate through the role nouns
    for role_noun_gender in ["neutral", "feminine", "masculine"]:
        for role_noun_label in role_nouns["neutral"]:
            # Iterate through the names
            for name_gender in names:
                for name in names[name_gender]:
                    # Generate the prompt sentence
                    role_noun = role_nouns[role_noun_gender][role_noun_label]
                    determiner = "an" if role_noun in AN_NOUNS else "a"
                    sentence = config["sentence_format"].format(name, determiner, role_noun)
                    
                    add_prompts(
                        data=data,
                        name=name,
                        name_gender=name_gender,
                        role_noun=role_noun,
                        role_noun_gender=role_noun_gender,
                        role_noun_label=role_noun_label,
                        sentence=sentence,
                        task_wording=config["task_wording"])

    df = pd.DataFrame(data)
    df.index.name = "index"
    output_path = f"{output_dir}/stimuli.csv"
    df.to_csv(output_path)

    # generate_prompt_summary_sheet(
    #     name, item_label, output_path, config["ways_of_asking"], config["contexts"])


def main(config_path):
    # Load the config
    config = load_json(config_path)
    config_dir = "/".join(config_path.split("/")[:-1])
    
    # Run the correct main function for the domain
    if config["domain"] == "role_nouns":
        main_role_nouns(config, config_dir)
    else:
        raise ValueError(f"Domain type not supported: {config['domain']}")
    

if __name__ == "__main__":
    main("experiment/config.json")