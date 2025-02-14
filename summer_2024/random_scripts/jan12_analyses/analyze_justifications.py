"""Assess the frequency of different keywords/themes in justifications."""

import collections
import itertools
import pandas as pd
import re


ANNOTATED_JUSTIFICATIONS_PATH = "random_scripts/jan12_analyses/20250207-revisions_project-justifications_sample_100-annotated.csv"

# FOR LATER
TRAIN_PATH = "random_scripts/jan12_analyses/train.csv"
TEST_PATH = "random_scripts/jan12_analyses/test.csv"


CONFIG_PATHS = [
        "analyses/piloting_jan1/improve/config.json",
        "analyses/piloting_jan1/improve_if_needed/config.json",
        "analyses/piloting_jan1/revise/config.json",
        "analyses/piloting_jan1/revise_if_needed/config.json"
]

KEYWORDS_OUTPUT_PATH = "random_scripts/jan12_analyses/keywords.csv"


def clean_keywords(keyword_list):
    # remove things in brackets or parens
    keyword_list = re.sub("\(.*\)", "", keyword_list).strip()
    keyword_list = re.sub("\[.*\]", "", keyword_list).strip()

    # Remove things preceded by colons + other irrelevant words
    keyword_list = keyword_list.replace("NEG:", "")
    keyword_list = keyword_list.replace("POS:", "")
    keyword_list = keyword_list.replace("GEN:", "")
    keyword_list = keyword_list.replace("etc", "")
    keyword_list = keyword_list.replace(";", ",")

    # split and strip whitespace
    keyword_list = keyword_list.split(",")
    keyword_list = [item.strip() for item in keyword_list]

    return keyword_list


def load_justifications(config_paths):
    # Load justifications into a single spreadsheet
    df = pd.DataFrame()
    for config_path in config_paths:
        justifications_processed_path = config_path.replace("config.json", "sbert_input_for_justifications.csv")
        curr_df = pd.read_csv(justifications_processed_path)
        curr_df["prompt_wording"] = config_path.split("/")[-2]
        df = pd.concat([df, curr_df])
    return df


def analyze_keywords():

    # load keywords from Barend's annotations
    df = pd.read_csv(ANNOTATED_JUSTIFICATIONS_PATH)
    keywords = list(df["comments"])
    keywords = list(itertools.chain.from_iterable([
        clean_keywords(keyword_list) for keyword_list in keywords]))
    keyword_counter = collections.Counter(keywords)
    selected_kws = keyword_counter.most_common(15)

    # load justifications
    justifications_df = load_justifications(CONFIG_PATHS)

    # assess frequency of keywords in justifications
    for kw, _ in selected_kws:
        justifications_df[f"KW_{kw}"] = [
            1 if re.findall(r"\b" + kw + r"\b", curr_justification) else 0
            for curr_justification in justifications_df["sbert_strings"]
        ]

    # Compute aggregated stats by condition
    cols = ["task_wording", "role_noun_gender"] + [f"KW_{kw}" for kw, _ in selected_kws]
    justifications_df = justifications_df[cols]
    output_df = justifications_df.groupby(["task_wording", "role_noun_gender"]).mean()
    output_df = output_df.reindex(
        [
            'unknown gender',
            'gender declaration nonbinary', 'gender declaration woman', 'gender declaration man',
            'pronoun declaration they/them', 'pronoun declaration she/her', 'pronoun declaration he/him',
            'pronoun usage their', 'pronoun usage her', 'pronoun usage his'
        ],
        level='task_wording')
    output_df = output_df.reindex(['neutral', 'feminine', 'masculine'], level='role_noun_gender')

    # Add count column to output_df
    count_df = justifications_df.groupby(["task_wording", "role_noun_gender"]).count()
    output_df["count"] = count_df["KW_inclusive"] # Could be any keyword

    output_df.to_csv(KEYWORDS_OUTPUT_PATH)


if __name__ == "__main__":
    print("running analysis")
    analyze_keywords()
