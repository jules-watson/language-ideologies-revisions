"""
Generate the stimuli file of prompts to be queried by LLMs.

This file is inspired from part_1_stimuli.py,
which can be found at: https://github.com/juliawatson/language-ideologies-2024/blob/main/fall_2023_main/part_1_stimuli.py

Author: Raymond Liu
Date: Jun 2024
"""
import ast
import collections
import pandas as pd

from common import load_json, load_csv, gender_list_dict
from constants import AN_NOUNS, EXPERIMENT_PATH


def add_prompts(data, 
                role_noun, role_noun_gender, role_noun_original_gender, role_noun_is_substituted, role_noun_set,
                sentence, sentence_format,
                task_wording):
    """
    Add one row to the stimuli.csv file.
    """
    for task_wording_label, task_wording in task_wording.items():
        data["role_noun"].append(role_noun)
        data["role_noun_gender"].append(role_noun_gender)
        data["role_noun_original_gender"].append(role_noun_original_gender)
        data["role_noun_is_substituted"].append(role_noun_is_substituted)
        data["role_noun_set"].append(role_noun_set)

        data["sentence"].append(sentence)
        data["sentence_format"].append(sentence_format)

        data["task_wording"].append(task_wording_label)

        final_sentence = f"{task_wording} {sentence}"
        data["prompt_text"].append(final_sentence)



def main_role_nouns(config, output_dir):
    """
    Create the stimuli file for role nouns.
    """
    role_nouns = load_json(config["role_nouns"]) 
    about_me_data = pd.read_csv(config["sentence_data"])

    gender_list = gender_list_dict[len(role_nouns[0])]

    data = collections.defaultdict(list)

    # Iterate through the role nouns
    for role_noun_set in role_nouns:
        for i in range(len(role_noun_set)):  
            # choose a specific role noun variant within a role noun set
            role_noun = role_noun_set[i] 
            matching_rows = about_me_data[about_me_data['filtered_roles_data'].str.contains(f"'{role_noun}'", case=False, na=False)]
            selected_rows = matching_rows.sample(min(10, len(matching_rows)))

            for index, row in selected_rows.iterrows():
                sentence = row['sentence']

                # Obtain the corresponding information (i.e. starting and ending index in sentence) of the role noun
                roles_info = ast.literal_eval(row['filtered_roles_data'])  
                obtained_role_noun, start_id, end_id = next((role_info for role_info in roles_info if role_info[0] == role_noun), None)
                assert(obtained_role_noun == role_noun)

                for j in range(len(role_noun_set)): 
                    # iterate through substitutions
                    sentence_format = sentence[:start_id] + '[ROLE NOUN]' + sentence[end_id:]
                    new_sentence = sentence[:start_id] + role_noun_set[j] + sentence[end_id:]

                    add_prompts(
                        data=data,
                        role_noun=role_noun_set[j],
                        role_noun_gender=gender_list[j],
                        role_noun_original_gender=gender_list[i],
                        role_noun_is_substituted=(i!=j),
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
        main_role_nouns(config, config_dir)
    else:
        raise ValueError(f"Domain type not supported: {config['domain']}")
    

if __name__ == "__main__":
    main(f"{EXPERIMENT_PATH}/config.json")