"""Assess the frequency of different keywords/themes in justifications."""

import collections
import itertools
import os
import pandas as pd
import re
import sklearn.metrics
from sklearn.model_selection import train_test_split


CONFIG_PATHS = [
        "analyses/piloting_jan1/improve/config.json",
        "analyses/piloting_jan1/improve_if_needed/config.json",
        "analyses/piloting_jan1/revise/config.json",
        "analyses/piloting_jan1/revise_if_needed/config.json"
]

THEME_WORDS_PATH = "random_scripts/mar1_analyses/theme_words.csv"

OUTPUT_PATH = "random_scripts/mar1_analyses/{}.csv"


def load_justifications(config_paths, revised_to=None, starting_variants=None, conditions=None):
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

    # filter based on the role noun in the sentence fed into the model
    if isinstance(starting_variants, list):
        df = df[df["role_noun_gender"].isin(starting_variants)]

    # filter to select particular conditions
    if isinstance(conditions, list):
        df = df[df["task_wording"].isin(conditions)]

    implicit_conditions = [
        "unknown gender",
        "pronoun usage their",
        "pronoun usage her",
        "pronoun usage his"
    ]
    df["implicit_explicit"] = [
        "implicit" if condition_name in implicit_conditions else "explicit" 
        for condition_name in df["task_wording"]]
    
    return df


def contains_words(justification, words):
    words_str = "|".join([f"({word})" for word in words])
    if re.search(r"\b(" + words_str + r")\b", justification, flags=re.IGNORECASE):
        return True
    return False


def analyze_justifications(
        description, themes, groupby_col, 
        revised_to=None, starting_variants=None, conditions=None):
    """
    method: str in {"keywords", "categories"}
    """

    # load justifications
    justifications_df = load_justifications(
        CONFIG_PATHS, 
        revised_to=revised_to, 
        starting_variants=starting_variants, 
        conditions=conditions)

    # load themes
    themes_df = pd.read_csv(THEME_WORDS_PATH)
    themes_df["name"] = [item.split("\t")[0] for item in themes_df["seed_set"]]
    themes_df = themes_df.set_index("name")

    heuristic_functions = {
        theme: lambda just, theme=theme: contains_words(just, words=themes_df.loc[theme]["theme_words"].split("\t"))
        for theme in themes
    }
    for category, heuristic_fn in heuristic_functions.items():
        justifications_df[f"cat_{category}"] = justifications_df["sbert_strings"].apply(heuristic_fn)
    analysis_cols = [f"cat_{category}" for category in heuristic_functions]

    # Compute aggregated stats by condition
    cols = [groupby_col] + analysis_cols
    justifications_df = justifications_df[cols]
    output_df = justifications_df.groupby([groupby_col]).mean()

    if output_df.index.name == "role_noun_gender":
        output_df = output_df.reindex(['neutral', 'feminine', 'masculine'], level='role_noun_gender')
    elif output_df.index.name == "task_wording":
        output_df = output_df.reindex(
            ['gender declaration nonbinary', 'gender declaration woman', 'gender declaration man'], 
            level='task_wording')


    # Add count column to output_df
    count_df = justifications_df.groupby([groupby_col]).count()
    output_df["count"] = count_df[analysis_cols[0]] # Could be any column

    output_df.to_csv(OUTPUT_PATH.format(f"{description}-{revised_to}"))


if __name__ == "__main__":
    # analyze_justifications(method="keywords", revised_to="alternative_wording")

    variants_themes = [
       "exclusionary", "modern", "professional", "common", "natural"
    ]
    analyze_justifications(
        description="variants_alt_wording",
        themes=variants_themes,
        groupby_col="role_noun_gender",
        revised_to="alternative_wording")
    
    gender_themes = [
        "exclusionary", "authentic", 
        "sexist", "professional",
        "modern"
    ]
    analyze_justifications(
        description="gender_neutralization",
        themes=gender_themes,
        groupby_col="task_wording",
        revised_to="neutral",
        starting_variants=["masculine", "feminine"],
        conditions=['gender declaration nonbinary', 'gender declaration woman', 'gender declaration man']
    )

    implicit_explicit_themes = [
        "exclusionary", "modern", "professional"
    ]
    analyze_justifications(
        description="implicit_explicit_neutralization",
        themes=implicit_explicit_themes,
        groupby_col="implicit_explicit",
        revised_to="neutral",
        starting_variants=["masculine", "feminine"],
    )


# ,seed_set,theme_words
# 0,exclusionary	inclusive,exclusionary	inclusive	exclusive	limiting	encompassing	welcoming	universal	outdated	problematic	standardized	unnecessary	acceptable	generic	cisnormative	restrictive
# 1,modern	traditional	outdated	contemporary,contemporary	modern	traditional	outdated	dated	sophisticated	refined	standardized	polished	archaic	rugged	regional	conventional	unconventional	informal
# 2,professional,professional	polished	technical	specialized	diplomatic	qualified	authoritative	rugged	confident	serious	occupational	formal	standardized	charitable	casual
# 3,common	standard,standard	common	preferred	traditional	acceptable	standardized	conventional	uncommon	correct	typical	used	consistent	formal	refined	universal
# 4,natural	fluid	clunky	awkward,fluid	awkward	natural	sophisticated	refined	smooth	polished	unconventional	ambiguous	straightforward	forced	repetitive	authoritative	playful	diplomatic
# 5,authentic,authentic	inspiring	authoritative	reliable	credible	honest	diplomatic	polished	meaningful	confident	intentional	engaging	welcoming	professional	playful
# 6,sexist,sexist	gendered	exclusionary	stereotypical	outdated	cisnormative	heteronormative	sensationalized	deceitful	male	masculine	provocative	confrontational	personalized	dismissive

