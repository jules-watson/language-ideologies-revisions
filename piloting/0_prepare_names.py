"""
Prepare names to be used for prompting.

This file is a modified version of part_0_prepare_names_sentences.py,
which can be found at: https://github.com/juliawatson/language-ideologies-2024/blob/main/fall_2023_main/part_0_prepare_names_sentences.py

Author: Raymond Liu
Date: Jun 2024
"""

import csv
import numpy as np
import pandas as pd
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



if __name__ == "__main__":
    prepare_names()
    pass