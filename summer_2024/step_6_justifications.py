"""TODO - add description

Built on examples here: https://radimrehurek.com/gensim/models/ldamodel.html
"""

import collections
import csv
from gensim.corpora.dictionary import Dictionary
from gensim.models import LdaModel
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np
import os
import pandas as pd
import re
from scipy.stats import fisher_exact
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples, silhouette_score
import tqdm
import warnings

from common import load_csv, load_json
from constants import EXPERIMENT_PATH, MODEL_NAMES
from visualizations import stacked_grouped_bar_graph


STOP_WORDS = set(stopwords.words('english'))
N_TOPICS = 10

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from common import load_json

from matplotlib.colors import ListedColormap
from constants import EXPERIMENT_PATH, MODEL_NAMES


def load_revisions(config_path, model_names):
    dirname = "/".join(config_path.split("/")[:-1])

    revisions_df = pd.DataFrame()
    for model_name in model_names:
        curr_revision_path = f"{dirname}/{model_name}/revision.csv"
        curr_revision_df = load_csv(curr_revision_path)
        curr_revision_df["model_name"] = model_name

        # Filter to select rows that were revised
        curr_revision_df = curr_revision_df[(curr_revision_df['variant_removed'] == True)]

        if any(pd.isna(curr_revision_df["justification"])):
            warnings.warn(f"Dropping nan justifications for model = {model_name}")
            curr_revision_df = curr_revision_df.dropna(subset=["justification"])

        # Add curr_revision_df to the end of revisions_df
        revisions_df = pd.concat([revisions_df, curr_revision_df])

    return revisions_df


def contains_any(sentence, role_noun_variants):
    for variant in role_noun_variants:
        if variant in sentence:
            return True
    return False


def mask_quoted_strings(sentence, role_noun_set):

    # Replace phrases in quotes with [MASK]
    re_quoted_phrase = re.compile(r"(\"[^\"]*\")|((^|\s)\'[^\']*\')")
    sentence = re_quoted_phrase.sub(" [MASK] ", sentence)

    # Replace any role nouns from the role noun set with [MASK]
    re_role_nouns = re.compile("[\"\']?(" + "|".join(role_noun_set) + ")[\"\']?", re.IGNORECASE)
    sentence = re_role_nouns.sub(" [MASK] ", sentence)

    # Replace multiple whitespace tokens with a space
    re_combine_whitespace = re.compile(r"\s+")
    sentence = re_combine_whitespace.sub(" ", sentence).strip()

    return sentence


def get_revisions_string_for_sbert_embedding(justification_string, role_noun_set):
    # Select only the sentences pertaining to the role noun set
    relevant_sentences = [
        sent for sent in sent_tokenize(justification_string)
        if contains_any(sent.lower(), role_noun_set)]

    # mask the quoted strings of the relevant sentences
    relevant_sentences = [mask_quoted_strings(sent, role_noun_set) for sent in relevant_sentences]

    return " ".join(relevant_sentences)


def get_sbert_embeddings(revisions_df, batch_size=16):
    model = SentenceTransformer('sentence-transformers/stsb-bert-base')
    embeddings = []    
    
    # encode each batch
    sentences = list(revisions_df["sbert_strings"])
    for i in tqdm.trange(len(sentences) // batch_size + 1):
        batch = list(sentences[i * batch_size : (i + 1) * batch_size])
        embeddings.append(model.encode(batch))
    embeddings = np.concatenate(embeddings)    

    assert len(embeddings) == len(revisions_df)
    return embeddings


def select_n_components(explained_variance_ratio, threshold):
    curr_variance_explained = 0
    for i in range(len(explained_variance_ratio)):
        curr_variance_explained += explained_variance_ratio[i]
        if curr_variance_explained >= threshold:
            return i + 1
    raise ValueError("Threshold not reached - not enough components")


def cluster_sbert_embeddings(X, min_clust=2, max_clust=10):
    # Apply pca before clustering; select n_components to explain 90% of the variance
    pca = PCA(n_components=X.shape[1], random_state=10)
    X_reduced = pca.fit_transform(X)
    n_components = select_n_components(pca.explained_variance_ratio_, 0.9)
    print(f"PCA: selected n_components={n_components}")
    X_reduced = X_reduced[:, :n_components]

    # Select the cluster with the highest silhouette score
    best_n_clusters = None
    best_silhouette_score = -1
    best_clustering = None
    for n_clusters in range(min_clust, max_clust + 1):
        clusterer = KMeans(n_clusters=n_clusters, random_state=10, n_init=10)
        cluster_labels = clusterer.fit_predict(X)

        silhouette_avg = silhouette_score(X_reduced, cluster_labels)
        print(
            "For n_clusters =",
            n_clusters,
            "The average silhouette_score is :",
            silhouette_avg,
        )

        if silhouette_avg > best_silhouette_score:
            best_n_clusters = n_clusters
            best_silhouette_score = silhouette_avg
            best_clustering = cluster_labels

    print(f"best n_clusters = {best_n_clusters}")

    return best_clustering


def get_cluster_to_vocab(cluster_df):
    result = collections.defaultdict(collections.Counter)
    cluster_df["sbert_strings_tokenized"] = cluster_df["sbert_strings"].apply(word_tokenize)
    for _, row in  cluster_df.iterrows():
        result[row["cluster"]].update(collections.Counter(row["sbert_strings_tokenized"]))
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

            cluster_word_assocations[f"cluster_{cluster}"].append(log_p)
    result = pd.DataFrame(cluster_word_assocations)
    result = result.set_index("word")
    return result


def get_cluster_keywords(config_path, cluster_df, min_frequency=100, n_keywords=40):
    """
    min_frequency is the minimum frequency per million words
    """
    # Create a dict mapping cluster ids to Counter vocab objects
    cluster_to_vocab = get_cluster_to_vocab(cluster_df)

    # identify words with a frequency greater than min_frequency (across clusters)
    vocab = collections.Counter()
    for curr_vocab in cluster_to_vocab.values():
        vocab.update(curr_vocab)
    threshold = (min_frequency * sum(vocab.values())) / 1000000
    vocab = {k: v for k, v in vocab.items() if v >= threshold}

    # store words associated with each cluster
    cluster_word_associations = compute_cluster_word_associations(
        cluster_to_vocab, vocab, alpha=1e-250)

    cluster_keywords_list = []
    for cluster in sorted(set(cluster_df["cluster"])):
        cluster_keywords_list.append({
            "cluster": cluster,
            "keywords": "\t".join(list(cluster_word_associations[f"cluster_{cluster}"].nlargest(n_keywords).index))
        })
    cluster_keywords_df = pd.DataFrame(cluster_keywords_list)
    output_path = config_path.replace("config.json", "cluster_keywords.csv")
    cluster_keywords_df.to_csv(output_path)


def get_visualization_df(df, prompt_wording, n_clusters):
    # n_clusters = len(set(df["cluster"]))
    df_list = []
    for gender_label in ['neutral', 'masculine', 'feminine']:
        gender_df = df[df["role_noun_gender"] == gender_label]
        curr_row_dict = {
            "prompt_wording": prompt_wording,
            "starting_variant": gender_label
        }
        cluster_counter = collections.Counter(list(gender_df["cluster"]))
        for cluster_i in range(n_clusters):
            curr_row_dict[f"cluster_{cluster_i}"] = cluster_counter.get(cluster_i, 0) / len(gender_df)
        df_list.append(curr_row_dict)
    
    result_df = pd.DataFrame(df_list)
    return result_df


def visualize_topics_by_llm(config_path, cluster_df):
    config = load_json(config_path)
    dirname = "/".join(config_path.split("/")[:-1])

    task_wording_dict = config['task_wording']
    n_clusters = len(set(cluster_df["cluster"]))

    for prompt_wording in task_wording_dict.keys():
        prompt_df = cluster_df[cluster_df['task_wording'] == prompt_wording]

        model_dfs = [prompt_df[prompt_df["model_name"] == model_name] for model_name in MODEL_NAMES]
        visualization_dfs = [
            get_visualization_df(df, prompt_wording, n_clusters) 
            for df in model_dfs]
        
        # revision rates plot for the current prompt
        escaped_prompt_wording = prompt_wording.replace('/', ' ')        
        output_path = f'{dirname}/justification_bar_graph_{escaped_prompt_wording}.png'
        print(output_path)
        print(visualization_dfs)
        stacked_grouped_bar_graph(
            data_frames=visualization_dfs,
            output_path=output_path,
            prompt_wording=prompt_wording,
            label_col="starting_variant")


def main(config_path):
    # Load and prepare data for sbert
    justifications_processed_path = config_path.replace("config.json", "sbert_input_for_justifications.csv")
    if os.path.exists(justifications_processed_path):
        justifications_processed_df = pd.read_csv(justifications_processed_path)
    else:
        justifications_processed_df = load_revisions(config_path, MODEL_NAMES)
        justifications_processed_df["sbert_strings"] = [
            get_revisions_string_for_sbert_embedding(row["justification"], eval(row["role_noun_set"]))
            for _, row in justifications_processed_df.iterrows()]

        num_rows_no_sbert_string = sum(justifications_processed_df["sbert_strings"] == "")
        if num_rows_no_sbert_string > 0:
            warnings.warn(f"Dropping n={num_rows_no_sbert_string} rows where sbert input strings were empty")
            justifications_processed_df = justifications_processed_df[justifications_processed_df["sbert_strings"] != ""]

        justifications_processed_df.to_csv(justifications_processed_path)

    # get sbert embeddings
    embeddings_path = config_path.replace("config.json", "sbert_embeddings.csv")
    if os.path.exists(embeddings_path):
        sbert_embeddings = np.genfromtxt(embeddings_path, delimiter=',')
    else:
        sbert_embeddings = get_sbert_embeddings(justifications_processed_df)
        np.savetxt(embeddings_path, sbert_embeddings, delimiter=",")

    assert len(sbert_embeddings) == len(justifications_processed_df)

    # Cluster sbert embeddings
    cluster_labels = cluster_sbert_embeddings(sbert_embeddings)
    justifications_processed_df["cluster"] = list(cluster_labels)
    cluster_keywords = get_cluster_keywords(config_path, justifications_processed_df)
    visualize_topics_by_llm(config_path, justifications_processed_df)


if __name__ == "__main__":
    main(f"{EXPERIMENT_PATH}/config.json")
