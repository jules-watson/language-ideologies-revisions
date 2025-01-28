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
import pandas as pd


def load_wording_to_keywords(wordings, keywords_format_str, n_clusters):
    """Returns a dict mapping wording [str] -> list of list.

    Each inner list in the values corresponds to the keywords associated 
    with a single cluster.
    """
    wording_to_keywords = {}
    for wording in wordings:
        curr_path = keywords_format_str.format(wording, n_clusters)
        curr_df = pd.read_csv(curr_path)
        wording_to_keywords[wording] = [keyword_set.split("\t") for keyword_set in curr_df["keywords"]]
    return  wording_to_keywords


def get_cluster_alignment_matrix(keywords_a, keywords_b):

    assert len(keywords_a) == len(keywords_b)
    n_clusters = len(keywords_a)

    result = np.zeros((n_clusters, n_clusters))

    for i in range(n_clusters):
        for j in range(n_clusters):
            curr_keyset_a = set(keywords_a[i])
            curr_keyset_b = set(keywords_b[j])

            curr_union = curr_keyset_a.union(curr_keyset_b)
            curr_intersection = curr_keyset_a.intersection(curr_keyset_b)

            curr_jaccard_distance = len(curr_intersection) / len(curr_union)

            result[i, j] = curr_jaccard_distance
    
    return result


def get_cluster_mapping(cluster_alignment_matrix):

    # populate_valid indices
    assert cluster_alignment_matrix.shape[0] == cluster_alignment_matrix.shape[1]
    n_clusters = cluster_alignment_matrix.shape[0]
    valid_indices = list(itertools.product(range(n_clusters), range(n_clusters)))

    mapping = []
    while valid_indices:

        # iterate over valid indices and find the max jaccard score
        max_position = valid_indices[0]
        max_value = cluster_alignment_matrix[max_position]
        for i, j in valid_indices:
            if cluster_alignment_matrix[i, j] > max_value:
                max_position = (i, j)
                max_value = cluster_alignment_matrix[i, j]

        # update mapping
        mapping.append((max_position[0], max_position[1]))

        # update valid_indices
        valid_indices = [
            item for item in valid_indices 
            if item[0] != max_position[0] and item[1] != max_position[1]]

    return mapping


def get_cluster_alignment_score(cluster_alignment_matrix, cluster_mapping):
    jaccard_scores = []
    for curr_i, curr_j in cluster_mapping:
        jaccard_scores.append(cluster_alignment_matrix[curr_i, curr_j])
    return np.mean(jaccard_scores)


def compute_keyword_alignment(keywords_a, keywords_b):
    """

    keywords_a: list of list, where each inner list is the set of 
                keywords associated with a single cluster
    keywords_b: list of list, where each inner list is the set of 
                keywords associated with a single cluster

    precondition: len(keywords_a) == len(keywords_b)

    returns:
        cluster_alignment_matrix: a matrix of len(keywords_a) x len(keywords_b).
            Cell i, j indicates the jaccard index of keywords associated with
            cluster i in keywords_a and cluster j in keywords_b.
        cluster_mapping: a list[tuple] indicating what clusters correspond to each other
            in clusters a and b. The list contains tuples (i, j) where i is a cluster 
            id in keywords_a and j is a cluster id in keywords_b.
        cluster_alignment_score: the average jaccard index for the clusters that
            map to each other in the cluster mapping.
    """
    # generate cluster alignment matrix
    cluster_alignment_matrix = get_cluster_alignment_matrix(keywords_a, keywords_b)

    # determine the cluster mapping
    cluster_mapping = get_cluster_mapping(cluster_alignment_matrix)

    # compute the cluster alignment score
    cluster_alignment_score = get_cluster_alignment_score(
        cluster_alignment_matrix, cluster_mapping)

    return cluster_alignment_matrix, cluster_mapping, cluster_alignment_score


def main(wordings, keywords_format_str, n_clusters):
    # load dictionary mapping wordings to keywords
    wording_to_keywords = load_wording_to_keywords(wordings, keywords_format_str, n_clusters)
    output_dir = "/".join(keywords_format_str.split("/")[:-2])
    
    wordings_matrix = np.ones([len(wordings), len(wordings)])
    # for each combination of wordings
    for wording_a, wording_b in itertools.combinations(wordings, 2):

        # compute cluster alignments, based on keywords
        cluster_alignment_matrix, cluster_mapping, cluster_alignment_score = compute_keyword_alignment(
            wording_to_keywords[wording_a],
            wording_to_keywords[wording_b]
        )

        # save cluster_alignment_matrix and cluster_mapping
        cluster_alignment_output_path = f"{output_dir}/{n_clusters}_cluster_alignment_matrix_{wording_a}_{wording_b}.csv"
        np.savetxt(cluster_alignment_output_path, cluster_alignment_matrix, delimiter=",", fmt="%.4f")

        cluster_mapping_output_path = f"{output_dir}/{n_clusters}_cluster_alignment_mapping_{wording_a}_{wording_b}.csv"
        np.savetxt(cluster_mapping_output_path, np.array(cluster_mapping), delimiter=",", fmt="%d")

        # update wordings_matrix
        wording_a_index = wordings.index(wording_a)
        wording_b_index = wordings.index(wording_b)
        wordings_matrix[wording_a_index][wording_b_index] = cluster_alignment_score
        wordings_matrix[wording_b_index][wording_a_index] = cluster_alignment_score
    
    # save wordings_matrix
    wordings_matrix_output_path = f"{output_dir}/{n_clusters}_wordings_matrix.csv"
    wordings_df = pd.DataFrame(wordings_matrix, columns=wordings, index=wordings)
    wordings_df.to_csv(wordings_matrix_output_path)


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