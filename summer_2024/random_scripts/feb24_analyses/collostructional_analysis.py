import collections
# from nltk.tokenize import word_tokenize
import spacy
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact


CONFIG_PATHS = [
        "analyses/piloting_jan1/improve/config.json",
        "analyses/piloting_jan1/improve_if_needed/config.json",
        "analyses/piloting_jan1/revise/config.json",
        "analyses/piloting_jan1/revise_if_needed/config.json"
]

OUTPUT_DIR = "random_scripts/feb24_analyses"


nlp = spacy.load("en_core_web_sm")


def load_justifications(config_paths, revised_to=None, starting_variants=None):
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
    
    return df


def get_group_to_vocab(group_to_justificaitons):
    result = collections.defaultdict(collections.Counter)

    for group, justifications in group_to_justificaitons.items():
        for justification in justifications:
            adjectives = [tok.text for tok in nlp(justification) if tok.pos_ == "ADJ"]
            result[group].update(collections.Counter(adjectives))

    return result


def compute_cluster_word_associations(cluster_to_vocab, vocab, alpha=1e-250):
    """
    alpha is the offset added to the p-value before taking the log
    """
    cluster_word_assocations = collections.defaultdict(list)
    for word in vocab:
        cluster_word_assocations["word"].append(word)

        for cluster in sorted(set(cluster_to_vocab.keys())):
    
            word_in_cluster = cluster_to_vocab[cluster].get(word, 0)
            word_elsewhere = vocab.get(word, 0) - word_in_cluster
            other_words_in_cluster = sum(cluster_to_vocab[cluster].values()) - word_in_cluster
            other_words_elsewhere = sum(vocab.values()) - sum(cluster_to_vocab[cluster].values()) - word_elsewhere

            # Set up categorical table
            table = np.array(
                [
                    [word_in_cluster,         word_elsewhere], 
                    [other_words_in_cluster,  other_words_elsewhere]
                ]
            )

            # Apply Fisher's test
            odds_r, p = fisher_exact(table)

            # Take log(p) for interpretability
            log_p = np.log(p + alpha)

            # Multiply by -1 or 1
            p_word_given_cluster = word_in_cluster / (word_in_cluster + other_words_in_cluster)
            p_word_otherwise = word_elsewhere / (word_elsewhere + other_words_elsewhere)
            if p_word_given_cluster > p_word_otherwise:
                log_p = -1 * log_p

            cluster_word_assocations[cluster].append(log_p)
    result = pd.DataFrame(cluster_word_assocations)
    result = result.set_index("word")
    return result


def run_collostructional_analysis(
        output_dir, description, group_to_justificaitons, min_frequency=20, n_keywords=40):
    """
    min_frequency is the minimum frequency per million words
    """
    # Create a dict mapping cluster ids to Counter vocab objects
    group_to_vocab = get_group_to_vocab(group_to_justificaitons)

    # identify words with a frequency greater than min_frequency (across clusters)
    vocab = collections.Counter()
    for curr_vocab in group_to_vocab.values():
        vocab.update(curr_vocab)
    threshold = (min_frequency * sum(vocab.values())) / 1000000
    vocab = {k: v for k, v in vocab.items() if v >= threshold}

    # compute and store words associated with each cluster
    cluster_word_associations = compute_cluster_word_associations(
        group_to_vocab, vocab, alpha=1e-250)

    cluster_keywords_list = []
    for cluster in sorted(set(group_to_justificaitons.keys())):
        cluster_keywords_list.append({
            "group": cluster,
            "keywords": ", ".join(list(cluster_word_associations[f"{cluster}"].nlargest(n_keywords).index))
        })
    cluster_keywords_df = pd.DataFrame(cluster_keywords_list)
    output_path = f"{output_dir}/{description}_keywords.csv"
    cluster_keywords_df.to_csv(output_path)


def compare_variants_revised_to_alternative_wording():
    """Compare themes when revising neut vs. masc vs. fem to alternative wording."""
    justifications_df = load_justifications(CONFIG_PATHS, revised_to="alternative_wording")
    
    neut_df = justifications_df[justifications_df["role_noun_gender"] == "neutral"]
    fem_df = justifications_df[justifications_df["role_noun_gender"] == "feminine"]
    masc_df = justifications_df[justifications_df["role_noun_gender"] == "masculine"]

    variant_to_justifications = {
        "neut": list(neut_df["sbert_strings"]),
        "fem": list(fem_df["sbert_strings"]),
        "masc": list(masc_df["sbert_strings"]),
    }

    run_collostructional_analysis(
        output_dir=OUTPUT_DIR,
        description="variants_alt_wording",
        group_to_justificaitons=variant_to_justifications,
    )


def compare_variants_revised_to_neutral():
    """Compare themes when revising masc vs. fem to neutral."""
    justifications_df = load_justifications(CONFIG_PATHS, revised_to="neutral")

    fem_df = justifications_df[justifications_df["role_noun_gender"] == "feminine"]
    masc_df = justifications_df[justifications_df["role_noun_gender"] == "masculine"]

    variant_to_justifications = {
        "fem": list(fem_df["sbert_strings"]),
        "masc": list(masc_df["sbert_strings"]),
    }

    run_collostructional_analysis(
        output_dir=OUTPUT_DIR,
        description="variants_neutral",
        group_to_justificaitons=variant_to_justifications,
    )


def compare_genders():
    """Compare themes when revising gendered to neutral 
    for nonbinary people vs. men vs. women."""
    justifications_df = load_justifications(
        CONFIG_PATHS,
        revised_to="neutral",
        starting_variants=["masculine", "feminine"])

    nonbinary_df = justifications_df[justifications_df["task_wording"] == "gender declaration nonbinary"]
    woman_df = justifications_df[justifications_df["task_wording"] == "gender declaration woman"]
    man_df = justifications_df[justifications_df["task_wording"] == "gender declaration man"]

    gender_to_justifications = {
        "nonbinary": list(nonbinary_df["sbert_strings"]),
        "woman": list(woman_df["sbert_strings"]),
        "man": list(man_df["sbert_strings"]),
    }

    run_collostructional_analysis(
        output_dir=OUTPUT_DIR,
        description="genders_rev_gendered_to_neut",
        group_to_justificaitons=gender_to_justifications,
    )


def compare_implicit_explicit():
    """Compare themes when revising gendered to neutral  
    for implicit vs. explicit conditions."""
    justifications_df = load_justifications(
        CONFIG_PATHS,
        revised_to="neutral",
        starting_variants=["masculine", "feminine"])
    
    implicit_conditions = [
        "unknown gender",
        "pronoun usage their",
        "pronoun usage her",
        "pronoun usage his"
    ]

    implicit_df = justifications_df[justifications_df["task_wording"].isin(implicit_conditions)]
    explicit_df = justifications_df[~justifications_df["task_wording"].isin(implicit_conditions)]

    implicit_explicit_to_justifications = {
        "implicit": list(implicit_df["sbert_strings"]),
        "explicit": list(explicit_df["sbert_strings"]),
    }

    run_collostructional_analysis(
        output_dir=OUTPUT_DIR,
        description="implicit_explict_rev_gendered_to_neut",
        group_to_justificaitons=implicit_explicit_to_justifications,
    )


if __name__ == "__main__":
    # compare_variants_revised_to_alternative_wording()
    # compare_variants_revised_to_neutral()
    compare_genders()
    compare_implicit_explicit()