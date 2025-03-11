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

from common import load_json, gender_list_dict
from constants import EXPERIMENT_PATH


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

            for _, row in selected_rows.iterrows():
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


def main_jan1(config, output_dir):
    sentences_df = pd.read_csv(config["sentence_data"])
    prompts_df = pd.DataFrame(list(config["task_wording"].keys()), columns=["task_wording"])

    df = pd.merge(sentences_df, prompts_df, how='cross')
    df["prompt_text"] = [
        f"{config['task_wording'][row['task_wording']]} {row['sentence']}" for _, row in df.iterrows()
    ]

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
        role_noun_original_gender = row["gender"]
        role_noun_neutral = row["role_noun_set"]
        role_noun_set = [
            role_nouns_df.loc[role_noun_neutral][curr_gender]
            for curr_gender in role_noun_genders
        ]

        # iterate through substitutions
        for role_noun_gender in role_noun_genders: 
            curr_role_noun = role_nouns_df.loc[role_noun_neutral][role_noun_gender]

            # make sentence template
            assert sentence.count(role_noun_original) == 1
            sentence_format = sentence.replace(role_noun_original, '[ROLE NOUN]')
            new_sentence = sentence_format.replace('[ROLE NOUN]', curr_role_noun)

            # This adds prompts for each task wording
            add_prompts(
                data=data,
                role_noun=curr_role_noun,
                role_noun_gender=role_noun_gender,
                role_noun_original_gender=role_noun_original_gender,
                role_noun_is_substituted=(role_noun_gender!=role_noun_original_gender),
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
        if ("stimuli_method" in config and 
            config["stimuli_method"] == "mar7_full_analysis"):
            main_mar7_full_analysis(config, config_dir)
        elif ("stimuli_method" in config and 
            config["stimuli_method"] == "jan1_piloting_pronouns_genders"):
            main_jan1(config, config_dir)
        else:
            # This is the default for now - it is for piloting and
            # downsamples sentences
            main_role_nouns(config, config_dir)
    else:
        raise ValueError(f"Domain type not supported: {config['domain']}")
    

if __name__ == "__main__":
    main(f"{EXPERIMENT_PATH}/config.json")