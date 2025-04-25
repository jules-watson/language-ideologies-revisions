"""Assess the frequency of different keywords/themes in justifications."""

import collections
import numpy as np
import pandas as pd
import re
from scipy.stats import chi2_contingency

from constants import EXPERIMENT_PATH


def load_justifications(config_path, revised_to=None, starting_variants=None, conditions=None):
    # Load justifications
    justifications_processed_path = config_path.replace("config.json", "bert_input_for_justifications.csv")
    df = pd.read_csv(justifications_processed_path)
    
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
        config_path, description, themes, groupby_col, 
        revised_to=None, starting_variants=None, conditions=None):
    """
    method: str in {"keywords", "categories"}
    """
    theme_words_path = config_path.replace("config.json", "theme_words.csv")

    # load justifications
    justifications_df = load_justifications(
        config_path=config_path, 
        revised_to=revised_to, 
        starting_variants=starting_variants, 
        conditions=conditions)

    # load themes
    themes_df = pd.read_csv(theme_words_path)
    themes_df = themes_df.set_index("name")
    theme_to_word_sets = {
        theme: list(set(row["theme_words"].split("\t") + row["seed_set"].split("\t")))
        for theme, row in themes_df.iterrows()
    }

    # Assess which jutifications contain the relevant keywords
    heuristic_functions = {
        theme: lambda just, theme=theme: contains_words(just, words=theme_to_word_sets[theme])
        for theme in themes
    }
    for category, heuristic_fn in heuristic_functions.items():
        justifications_df[category] = justifications_df["bert_strings"].apply(heuristic_fn)
    analysis_cols = [category for category in heuristic_functions]

    # Compute aggregated stats by condition
    cols = [groupby_col] + analysis_cols
    justifications_df = justifications_df[cols]
    output_df = justifications_df.groupby([groupby_col]).sum()

    if output_df.index.name == "role_noun_gender":
        output_df = output_df.reindex(['neutral', 'feminine', 'masculine'], level='role_noun_gender')
    elif output_df.index.name == "task_wording":
        output_df = output_df.reindex(
            ['gender declaration nonbinary', 'gender declaration woman', 'gender declaration man',
            'pronoun declaration they/them', 'pronoun declaration she/her', 'pronoun declaration he/him', 
            'pronoun usage their', 'pronoun usage her', 'pronoun usage his'], 
            level='task_wording')

    # Add count column to output_df
    count_df = justifications_df.groupby([groupby_col]).count()
    output_df["count"] = count_df[analysis_cols[0]] # Could be any column

    # Add proportion columns to output_df
    for col in output_df.columns:
        if col in analysis_cols:
            new_colname = "prop_" + col
            output_df[new_colname] = output_df[col] / output_df["count"]

    output_path = config_path.replace("config.json", f"{description}.csv")
    output_df.to_csv(output_path) 

    output_df = output_df[[col for col in output_df.columns if "prop_" in col]]
    output_df = output_df.T
    latex_output_path = config_path.replace("config.json", f"{description}.tex_table")
    output_df.to_latex(latex_output_path, float_format="%.2f") 


def get_pval_str(p, bonferroni_n):
    corrected_p = p * bonferroni_n

    if corrected_p <= 0.001:
        return "***"
    elif corrected_p <= 0.01:
        return "**"
    elif corrected_p <= 0.05:
        return "*"
    return "n.s."


def run_stats_tests(
        config_path, group1, group2, input_description, output_description=None, themes=None):
    """

       - config_path: path to config - indicates dir to to read and write to
       - group1: list[str] indicating conditions to test
       - input_description: indicates where to load counts from for test
       - output_description: indicates where to write test results to (defaults to input_description)
       - themes: list of theme names to run tests for (defaults to all themes)
    """
    if output_description is None:
        output_description = input_description

    data_path = config_path.replace("config.json", f"{input_description}.csv")
    df = pd.read_csv(data_path, index_col=0)

    if themes is None:
        themes = [theme_name for theme_name, row in df.T.iterrows() 
                  if not theme_name.startswith("prop_") and theme_name != "count"]
    bonferroni_n = len(themes)
    print(f"Bonferroni correction for: {bonferroni_n} themes")

    counts = df["count"]
    group1_total_count = int(sum([counts[group1_cond] for group1_cond in group1]))
    group2_total_count = int(sum([counts[group2_cond] for group2_cond in group2]))

    output_dict = collections.defaultdict(list)
    for theme_name, row in df.T.iterrows():
        if theme_name not in themes:
            continue

        group1_theme_count = int(sum([row[group1_cond] for group1_cond in group1]))
        group2_theme_count = int(sum([row[group2_cond] for group2_cond in group2]))

        # Set up categorical table
        table = np.array(
            [
                [group1_theme_count,    group1_total_count - group1_theme_count], 
                [group2_theme_count,    group2_total_count - group2_theme_count]
            ]
        )

        # Apply Fisher's test
        chi2, p, dof, _ = chi2_contingency(table)

        # Update output dict
        output_dict["theme"].append(theme_name)
        output_dict["chi2"].append(chi2)
        output_dict["p"].append(p)
        output_dict["n"].append(np.sum(table))
        output_dict["bonferroni_n"].append(bonferroni_n)
        output_dict["outcome"].append(get_pval_str(p, bonferroni_n))
        output_dict["dof"].append(dof)

    output_df = pd.DataFrame(output_dict)

    # write csv with all the info
    output_csv = config_path.replace("config.json", f"chisq_{output_description}.csv")
    output_df.to_csv(output_csv)

    # write latex with columns to include in the paper
    output_latex = config_path.replace("config.json", f"chisq_{output_description}.tex_table")
    output_df["prediction"] = ""
    output_df[["theme", "prediction", "outcome"]].to_latex(output_latex, index=False)



def main(config_path):

    variants_themes = [
       "inclusive", "modern", "professional", "standard", "natural"
    ]
    analyze_justifications(
        config_path,
        description="variants_alt_wording",
        themes=variants_themes,
        groupby_col="role_noun_gender",
        revised_to="alternative_wording")
    
    gender_themes = [
        "inclusive", "modern", "professional",
    ]
    analyze_justifications(
        config_path,
        description="gender_neutralization",
        themes=gender_themes,
        groupby_col="task_wording",
        revised_to="neutral",
        starting_variants=["masculine", "feminine"],
        conditions=[
            'gender declaration nonbinary', 'gender declaration woman', 'gender declaration man',
            'pronoun declaration they/them', 'pronoun declaration she/her', 'pronoun declaration he/him', 
            'pronoun usage their', 'pronoun usage her', 'pronoun usage his', 
        ]
    )

    # run stats tests for variants:
    # inclusive, modern, professional more for neutral
    # standard, natural more for masc and fem
    run_stats_tests(
        config_path,
        input_description="variants_alt_wording",
        group1=["neutral"],
        group2=["feminine", "masculine"]
    )

    # run stats tests for genders:
    # inclusive, professional, and modern for nonbinary vs. binary genders
    #  -> also do a sub-test showing more for women vs. men
    # modern and professional more for men and women
    #  -> also do a sub-test showning more for woman vs. men for professional
    run_stats_tests(
        config_path,
        group1=["gender declaration nonbinary"],
        group2=["gender declaration woman", "gender declaration man"],
        input_description="gender_neutralization",
        output_description="genders-gend-neut",
    )
    run_stats_tests(
        config_path,
        group1=["gender declaration woman"],
        group2=["gender declaration man"],
        input_description="gender_neutralization",
        output_description="genders-woman_man",
        themes=["inclusive", "professional"]
    )

    # run stats tests for explicit vs. implicit - pronoun declaration vs. pronoun usage
    # for the 3 gender themes ("inclusive", "modern", "professional")
    # how do we do bonferroni correction here?
    run_stats_tests(
        config_path,
        group1=["pronoun declaration they/them", "pronoun declaration she/her", "pronoun declaration he/him"],
        group2=["pronoun usage their", "pronoun usage her", "pronoun usage his"],
        input_description="gender_neutralization",
        output_description="implicit-explicit",
    )


if __name__ == "__main__":
    main(f"{EXPERIMENT_PATH}/config.json")