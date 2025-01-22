"""Compare clusterings based on keywords.

The goal of this analysis is assess how similar the clusters are across
different wordings and numbers of clusters. This will inform: 
    (a) what we can claim about our results' robustness across prompt wordings 
        (if the results are not robust we may need to re-assess our design)
    (b) what we can claim about our results robustness across clusters
        (if the results are not robust, we may need to find another - perhaps
        more supervised/targeted - approach for analyzing variation across
        our conditions).
"""

import itertools
import numpy as np 


def load_wording_to_keywords(wordings, keywords_format_str, n_clusters):
    # TODO
    pass

    # return  wording_to_keywords


def compute_keyword_alignment(keywords_a, keywords_b):
    # TODO
    pass

    # return cluster_alignment_matrix, cluster_alignment_score


def main(wordings, keywords_format_str, n_clusters):
    # load dictionary mapping wordings to keywords
    wording_to_keywords = load_wording_to_keywords(wordings, keywords_format_str, n_clusters)
    
    wordings_matrix = np.ones([len(wordings), len(wordings)])
    # for each combination of wordings
    for wording_a, wording_b in itertools.combinations(wordings):

        # compute cluster alignments, based on keywords
        cluster_alignment_matrix, cluster_alignment_score = compute_keyword_alignment(
            wording_to_keywords[wording_a],
            wording_to_keywords[wording_b]
        )

        # save cluster_alignment_matrix
        # TODO

        # update wordings_matrix
        wording_a_index = wordings.index(wording_a)
        wording_b_index = wordings.index(wording_b)
        wordings_matrix[wording_a_index][wording_b_index] = cluster_alignment_score
    
    # save wordings_matrix
    # TODO


if __name__ == "__main__":

    WORDINGS = [
        "improve",
        "improve_if_needed",
        "revise",
        "revise_if_needed"
    ]

    KEYWORDS_FORMAT_STR = "analyses/piloting_jan1/{}/{}_cluster_keywords.csv"

    CLUSTER_NUMBERS = [
        2,  # 2 clusters is optimal for revise_if_needed (updated sentences to match piloting_pronouns_genders)
        3,  # 3 clusters is optimal for improve / piloting_pronouns_genders (updated splitting)
        6,  # 6 clusters *was* optimal for revise_if_needed (original mismatched sentence set)
    ]

    for n_clusters in CLUSTER_NUMBERS:
        main(
            wordings=WORDINGS,
            keywords_format_str=KEYWORDS_FORMAT_STR,
            n_clusters=n_clusters
        )