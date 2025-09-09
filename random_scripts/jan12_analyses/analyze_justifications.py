"""Assess the frequency of different keywords/themes in justifications."""

import collections
import itertools
import os
import pandas as pd
import re
import sklearn.metrics
from sklearn.model_selection import train_test_split


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

OUTPUT_PATH = "random_scripts/jan12_analyses/{}.csv"


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


def load_justifications(config_paths, revised_to=None):
    # Load justifications into a single spreadsheet
    df = pd.DataFrame()
    for config_path in config_paths:
        justifications_processed_path = config_path.replace("config.json", "sbert_input_for_justifications.csv")
        curr_df = pd.read_csv(justifications_processed_path)
        curr_df["prompt_wording"] = config_path.split("/")[-2]
        df = pd.concat([df, curr_df])
    
    # These are all cases that were revised
    assert df["variant_removed"].all()

    # filter based on what response was revised to
    if isinstance(revised_to, str):
        if revised_to == "alternative_wording":
            df = df[df['variant_added'].isnull()]  # no other variant was added
        else:  # one of masculine, neutral, feminine
            df = df[df["variant_added"] == revised_to]
    
    return df


def contains_words(justification, words):
    words_str = "|".join([f"({word})" for word in words])
    return re.search(r"\b(" + words_str + r")\b", justification, flags=re.IGNORECASE)


def is_gender_specific(justification):
    if contains_words(justification, [
        "woman", "man", "men", "women", "female", "male", 
        "nonbinary",
        "reflect", "align", 
        "they/them", "she/her", "he/him"
        ]):
        return 1
    return 0


def is_inclusivity(justification):
    if contains_words(justification, ["inclusive", "gender-neutral", "neutrality"]):
        return 1
    return 0

def is_positive(justification):
    if contains_words(
            justification, 
            ['formal', 'professional', 
             # 'gender-neutral', 'inclusive', 
             'modern', 'accurate', 'common', 
             # 'gender-specific', 
             'engaging', 'technical', 'clarity', 
             #'informal', 
             'widely recognized',
            #  'professionalism', 
             # 'outdated',
             'conversational', 'commonly used', 'concise', 'commonly accepted',
             # 'gender', 'woman', 
             'dynamic', 
             # 'sense of experience and expertise',
             'suitable', 
             #'suggests active and positive engagement', 
             'active', 'positive'
             'efficiently communicates', 'appealing', 
             # 'gender appropriate', 'consistency', 'sense of agency', 
             # 'specific', 
             # 'non-professional', 'colloquial', 
             'universally understood', 
            #  'accuracy', 
             'standard', 'widely used',
             'approachable', 
             # 'less relatable', 'identifies as', 
             # 'appropriate', 
             # 'not widely recognized', 
             'appreciated', 'widely accepted', 'direct', 
             # 'provessional', 'pronouns', 'consie', 
             'succinct', 'polished', 
             # 'gendered', 'respectful', 'professionally accepted', 
             "professionally", 
             'widely understood', 
             # 'dynamic quality', 
             'enthusiasm', 'positivity', 'inviting', 'appeal']
        ):
        return 1
    return 0


def is_negative(justification):
    if contains_words(
            justification, 
            # ["informal", "outdated", "trying too hard", "non-professional"]
            ['informal', 
              # 'professional', 'formal', 'gender', 'not commonly used', 
              'trying too hard', 
              # 'specific', 
              'non-professional', 'colloquial', 
              # 'accurate', 'conversational', 'approachable', 
              'less relatable', 
              #'not widely recognized', 'appreciated', 'technical', 'widely accepted', 
              'outdated', #'commonly used', 'modern',
              'inconsistent']
        ):
        return 1
    return 0



def analyze_justifications(method, revised_to=None):
    """
    method: str in {"keywords", "categories"}
    """

    # load justifications
    justifications_df = load_justifications(CONFIG_PATHS, revised_to=revised_to)

    # assess frequency of keywords in justifications
    if method == "keywords":
        # load keywords from Barend's annotations
        df = pd.read_csv(ANNOTATED_JUSTIFICATIONS_PATH)
        keywords = list(df["comments"])
        keywords = list(itertools.chain.from_iterable([
            clean_keywords(keyword_list) for keyword_list in keywords]))
        keyword_counter = collections.Counter(keywords)
        selected_kws = keyword_counter.most_common(15)

        for kw, _ in selected_kws:
            justifications_df[f"KW_{kw}"] = [
                1 if re.findall(r"\b" + kw + r"\b", curr_justification) else 0
                for curr_justification in justifications_df["sbert_strings"]
            ]
        analysis_cols = [f"KW_{kw}" for kw, _ in selected_kws]
    else:
        assert method == "categories"
        
        heuristic_functions = {
            "gender-specific": is_gender_specific,
            "inclusivity": is_inclusivity, 
            "positive": is_positive,
            "negative": is_negative 
        }
        for category, heuristic_fn in heuristic_functions.items():
            justifications_df[f"cat_{category}"] = justifications_df["sbert_strings"].apply(heuristic_fn)
        analysis_cols = [f"cat_{category}" for category in heuristic_functions]

    # Compute aggregated stats by condition
    cols = ["task_wording", "role_noun_gender"] + analysis_cols
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
    output_df["count"] = count_df[analysis_cols[0]] # Could be any column

    output_str = method if revised_to is None else f"{method}-{revised_to}"
    output_df.to_csv(OUTPUT_PATH.format(output_str))


def run_heuristic_algorithm(
        mode="train",
        categories=["gender-specific", "inclusivity", "positive", "negative"]):
    """For training and testing heuristic algorithms for categorising justifications.
    mode: str in {'train', 'test'}
    """
    
    # Split train test (or load train/test)
    if os.path.exists(TRAIN_PATH) and os.path.exists(TEST_PATH):
        train_data = pd.read_csv(TRAIN_PATH)
        test_data = pd.read_csv(TEST_PATH)
    else:
        # Load justifications and categories from my annotations
        # categories are: gender-specific-JW, inclusivity-JW, positive-JW, negative-Jw
        df = pd.read_csv(ANNOTATED_JUSTIFICATIONS_PATH)
        df["gender-specific-JW"] = [
            int(isinstance(subcat, str) and "gender-specific" in subcat)
            for subcat in df['sub-category']]
        df["inclusivity-JW"] = [
            int(isinstance(subcat, str) and "inclusivity" in subcat)
            for subcat in df['sub-category']]
        
        # Split into train and test and save
        train_data, test_data = train_test_split(df, test_size=0.3)
        train_data.to_csv(TRAIN_PATH)
        test_data.to_csv(TEST_PATH)

    # Run heuristic algorithms for each category
    heuristic_functions = {
        "gender-specific": is_gender_specific,
        "inclusivity": is_inclusivity, 
        "positive": is_positive,
        "negative": is_negative 
    }
    for category in categories:
        category_col = category + "-JW"
        heuristic_fn = heuristic_functions[category]

        print(f"================ {category} ================")

        if mode == "train":
            y_pred = train_data["sbert_strings"].apply(heuristic_fn)
            y_true = train_data[category_col]

            # identify cases that don't match and print them out
            print("FALSE POSITIVES - included but shouldn't be")
            false_pos = train_data.loc[(y_pred == 1) & (y_true == 0)]
            print("\t" + "\n\t".join(false_pos["sbert_strings"]) + "\n")

            print("FALSE NEGATIVES - aren't included but should be")
            false_pos = train_data.loc[(y_pred == 0) & (y_true == 1)]
            print("\t" + "\n\t".join(false_pos["sbert_strings"]) + "\n")

        else: 
            assert mode == "test"
            y_pred = test_data["sbert_strings"].apply(heuristic_fn)
            y_true = test_data[category_col]

        # compute accuracy, precision, recall, f1
        print(f"Metrics for {category}")
        print('\tAccuracy: {:.4f}'.format(sklearn.metrics.accuracy_score(y_true, y_pred)))
        print('\tRecall: {:.4f}'.format(sklearn.metrics.recall_score(y_true, y_pred)))
        print('\tPrecision: {:.4f}'.format(sklearn.metrics.precision_score(y_true, y_pred)))
        print('\tF1 score: {:.4f}'.format(sklearn.metrics.f1_score(y_true, y_pred)))

        print("\n\n\n")


if __name__ == "__main__":
    # analyze_justifications(method="keywords")

    # run_heuristic_algorithm(mode="test")
    # analyze_justifications(method="categories")

    # analyze_justifications(method="keywords", revised_to="alternative_wording")
    # analyze_justifications(method="categories", revised_to="alternative_wording")

    analyze_justifications(method="keywords", revised_to="neutral")
    analyze_justifications(method="categories", revised_to="neutral")
