"""
Prepare names and sentences to be used.
Taken from https://github.com/juliawatson/language-ideologies-2024/blob/main/fall_2023_main/part_0_prepare_names_sentences.py
"""

import csv
import numpy as np
import pandas as pd
import random
import json


CAMILLIERE_DATA_PATH = "data/camilliere_stimuli.csv"
BABY_NAMES_DATA_PATH = "data/1998.txt"  # US social security baby names data from 1998


def load_names():
    # Load Camilliere names (which they sorted as gendered/non-gendered, based
    # on a norming study)
    camilliere_df = pd.read_csv(CAMILLIERE_DATA_PATH)
    gendered_names = list(set(camilliere_df[camilliere_df["cond"] == "gname"]["antecedent"]))
    non_gendered_names = list(set(camilliere_df[camilliere_df["cond"] == "ngname"]["antecedent"]))

    # Split the gendered names into masculine and feminine, using the US
    # social security baby names data from 1998
    baby_names_df = pd.read_csv(BABY_NAMES_DATA_PATH, names=["name", "sex", "count"])
    feminine_names, masculine_names = [], []
    for curr_name in gendered_names:
        curr_name_df = baby_names_df[baby_names_df["name"] == curr_name]
        fem_count = curr_name_df[curr_name_df["sex"] == "F"]["count"].item() if "F" in set(curr_name_df["sex"]) else 0
        masc_count = curr_name_df[curr_name_df["sex"] == "M"]["count"].item() if "M" in set(curr_name_df["sex"]) else 0
        if fem_count > masc_count:
            feminine_names.append(curr_name)
        else:
            masculine_names.append(curr_name)

        # Names in the gendered list should skew strongly towards M or F
        p_fem = fem_count / (fem_count + masc_count)
        assert p_fem < 0.2 or p_fem > 0.8, f"name={curr_name} p_fem={p_fem}"

    return non_gendered_names, feminine_names, masculine_names


def prepare_names(n_names=2, output_csv=False):
    non_gendered_names, feminine_names, masculine_names = load_names()

    # Sample n_names*2 gender-neutral names, including Alex
    alex = ["Alex"]
    ng_names_minus_alex = [curr_name for curr_name in non_gendered_names if curr_name not in alex]
    ng_sample = list(np.random.choice(ng_names_minus_alex, size=(n_names * 2) - 1, replace=False))
    ng_sample = ng_sample + alex

    # Sample n_names feminine and masculine names
    fem_sample = np.random.choice(feminine_names, size=n_names, replace=False)
    masc_sample = np.random.choice(masculine_names, size=n_names, replace=False)

    if output_csv:
        output_path = "data/names_sampled.csv"
        with open(output_path, "w") as f:
            csv_writer = csv.DictWriter(f, fieldnames=["group", "name"])
            for name_group, name_sample in {"neutral": ng_sample, "feminine": fem_sample, "masculine": masc_sample}.items():
                    for curr_name in name_sample:
                        csv_writer.writerow({
                            "group": name_group,
                            "name": curr_name
                        })
    else:
        output_path = "data/names_sampled.json"
        with open(output_path, "w") as f:
            data = {"neutral": ng_sample, "feminine": list(fem_sample), "masculine": list(masc_sample)}
            json.dump(data, f, indent=4)



def parse_camilliere_sentence_formats():
    """
    Format all sentences in camilliere_stimuli.csv to prepare for model querying and save to "stimuli/standard.csv"
    Generate masked form of sentence and add reference to determiner. 
    """
    data = pd.read_csv(CAMILLIERE_DATA_PATH)
    data = data.loc[data["cond"] == "ngname"]
    result = []
    for _, row in data.iterrows():
        curr_sentence = row["sentence"]
        curr_sentence = curr_sentence.replace(row["form"], "[FORM]")
        curr_sentence = curr_sentence.replace(row["antecedent"], "[NOUN]")
        curr_sentence = curr_sentence.replace(row["antecedent"], "[NOUN]")
        
        result.append({
            "itm": row["itm"],
            "sentence": curr_sentence,
            "form": row["form"]
        })

    result_df = pd.DataFrame(result)
    result_df.to_csv("data/camilliere_formats.csv", index=False)


def sample_forms():
    # Sample forms for the pilot. This loads from the original camilliere formats
    # file, which had a mistake for item #15 (It had "the laundromat attendant"
    # as the antecedent, rather than "the cowboy". This is becuase of a mistake
    # in the original Camilliere stimuli file, which only affected one of the 
    # versions of that item.)
    data = pd.read_csv("data/camilliere_formats-original.csv")
    forms = ["themselves", "them", "their", "they"]
    df = pd.DataFrame()
    for form in forms: 
        cur_data = data.loc[data["form"] == form]
        rand1 = random.randint(0, len(cur_data) - 1)
        form1 = cur_data.iloc[rand1-1:rand1]
        rand2 = random.randint(0, len(cur_data) - 1)
        while rand2 == rand1:
            rand2 = random.randint(0, len(cur_data) - 1)
        form2 = cur_data.iloc[rand2-1:rand2]
        df = pd.concat([df, form1, form2])
    df.to_csv("data/camilliere_subset.csv", index=False)


def prepare_singular_pronouns_sentences():
    parse_camilliere_sentence_formats()
    # sample_forms()   # Sample for pilot -- only run this once


if __name__ == "__main__":
    # prepare_singular_pronouns_sentences()
    prepare_names()
    pass