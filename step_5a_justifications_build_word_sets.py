import collections
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np
import os
import pandas as pd
import re
from sklearn.neighbors import NearestNeighbors
import spacy
from transformers import BertTokenizer, BertModel
import tqdm
import torch
import warnings

from common import load_csv, load_json
from constants import EXPERIMENT_PATH, MODEL_NAMES, GENDER_INFORMATION_CONDITIONS


nlp = spacy.load("en_core_web_sm")

# Load pre-trained model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')


def load_revisions(config_path, model_names, conditions=GENDER_INFORMATION_CONDITIONS):
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

    revisions_df = revisions_df[revisions_df["task_wording"].isin(conditions)]
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


def get_revisions_string_for_bert_embedding(justification_string, role_noun_set):
    # Select only the sentences pertaining to the role noun set
    relevant_sentences = [
        sent for sent in sent_tokenize(justification_string)
        if contains_any(sent.lower(), role_noun_set)]

    # mask the quoted strings of the relevant sentences
    relevant_sentences = [mask_quoted_strings(sent, role_noun_set) for sent in relevant_sentences]

    return " ".join(relevant_sentences)


def prepare_bert_inputs(config_path):
    # Load and prepare data for bert
    justifications_processed_path = config_path.replace("config.json", "bert_input_for_justifications.csv")
    if os.path.exists(justifications_processed_path):
        justifications_processed_df = pd.read_csv(justifications_processed_path)
    else:
        justifications_processed_df = load_revisions(config_path, MODEL_NAMES)
        justifications_processed_df["bert_strings"] = [
            get_revisions_string_for_bert_embedding(row["justification"], eval(row["role_noun_set"]))
            for _, row in justifications_processed_df.iterrows()]

        num_rows_no_bert_string = sum(justifications_processed_df["bert_strings"] == "")
        if num_rows_no_bert_string > 0:
            warnings.warn(f"Dropping n={num_rows_no_bert_string} rows where bert input strings were empty")
            justifications_processed_df = justifications_processed_df[justifications_processed_df["bert_strings"] != ""]

        justifications_processed_df.to_csv(justifications_processed_path)

    return justifications_processed_df


def extract_adjectives_from_justifications(justifications_df):
    result = collections.Counter()
    for _, row in tqdm.tqdm(justifications_df.iterrows()):
        justification = row["bert_strings"]
        adjectives = [tok.text.lower() for tok in nlp(justification) if tok.pos_ == "ADJ"]
        result.update(collections.Counter(adjectives))
    return result


def get_contextual_embeddings(justifications_df, adjectives):

    result = collections.defaultdict(list)
    for _, row in tqdm.tqdm(justifications_df.iterrows()):
        justification = row["bert_strings"]

        inputs = tokenizer(justification, return_tensors='pt')
        with torch.no_grad():
            outputs = model(**inputs)

        last_hidden_states = outputs.last_hidden_state
        tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

        cumulative_token = ""
        cumulative_embedding = []
        for token, embedding in zip(tokens, last_hidden_states[0]):

            #  This is a continuation (including some key hyphenated words)
            if token.startswith("##") or token == "-" or cumulative_token.endswith("-"):
                cumulative_token = cumulative_token + token.lstrip("##").lower()
                cumulative_embedding = cumulative_embedding + [embedding.numpy()]

            else:  # This is the beginning of a new word

                # Deal with the completed token
                if cumulative_token in adjectives:
                    cumulative_embedding = list(np.array(cumulative_embedding).mean(axis=0))
                    result["embedding"].append(cumulative_embedding)
                    result["adjective"].append(cumulative_token)

                # Start a new cumulative token
                cumulative_token = token.lower()
                cumulative_embedding = [embedding.numpy()]
                
        # We don't need to handle the case where the final token is part
        # of an adjective because the final token is always [SEP]
        assert cumulative_token.upper() == "[SEP]"

    return pd.DataFrame(result)


def generate_adj_embeddings(config_path, justifications_df):

    output_dir = "/".join(config_path.split("/")[:-1])

    # Step 1: get adjectives (~4 mins)
    adjectives = extract_adjectives_from_justifications(justifications_df)
    adjectives = set(adjectives)

    # Manually add hyphenated adjs identified by Barend's annotation as common keywords
    adjectives.add("gender-neutral")
    adjectives.add("gender-specific")
    adjectives.add("non-binary")

    # Step 2: generate contextual embeddings for each adjective occurrence
    # Will this take too much space? It's ok. 
    #  - for bert embeddings with 13590 data points, it takes 677 MB
    #  -> for bert embeddings with 24460 data points, it would take 1.2185 GB
    contextual_embeddings_output_path = f"{output_dir}/contextual_embeddings.csv"
    if os.path.exists(contextual_embeddings_output_path):
        contextual_embeddings = pd.read_csv(contextual_embeddings_output_path)
        contextual_embeddings["embedding"] = [eval(item) for item in contextual_embeddings["embedding"]]
    else:
        contextual_embeddings = get_contextual_embeddings(justifications_df, adjectives)
        contextual_embeddings.to_csv(contextual_embeddings_output_path)

    # Step 3: average embeddings per adjective and store them
    adj_embedding_output_path = f"{output_dir}/adj_embddings.csv"
    adj_embedding_dict = collections.defaultdict(list)
    for adj, adj_embeddings in contextual_embeddings.groupby("adjective"):
        mean_embedding = np.array(list(adj_embeddings["embedding"])).mean(axis=0)
        adj_embedding_dict["adjective"].append(adj)
        adj_embedding_dict["embedding"].append(list(mean_embedding))
    adj_embedding_df = pd.DataFrame(adj_embedding_dict)
    adj_embedding_df.to_csv(adj_embedding_output_path)


def build_theme_word_sets(seed_sets, config_path):

    output_dir = "/".join(config_path.split("/")[:-1])
    config = load_json(config_path)
    n = config["n_adj_neighbors"]

    # Step 1: load embeddings
    adj_embedding_path = f"{output_dir}/adj_embddings.csv"
    adj_embedding_df = pd.read_csv(adj_embedding_path)
    adj_embedding_df["embedding"] = [eval(item) for item in adj_embedding_df["embedding"]]
    adj_embedding_df = adj_embedding_df.set_index("adjective")

    # Step 2: Construct embeddings for each seed set
    seed_embeddings = []
    for seed_set_name, seed_set in seed_sets.items():
        curr_embeddings = adj_embedding_df.loc[list(seed_set)]
        seed_embedding = np.array(list(curr_embeddings["embedding"])).mean(axis=0)
        seed_embeddings.append(seed_embedding)
    seed_embeddings = np.array(seed_embeddings)

    # Step 3: Identify nearest neighbors
    X = np.array(list(adj_embedding_df["embedding"]))
    nbrs = NearestNeighbors(n_neighbors=n, algorithm='ball_tree').fit(X)
    _, indices = nbrs.kneighbors(seed_embeddings)
    
    output_dict = collections.defaultdict(list)
    assert len(seed_sets) == len(indices)
    for seed_set_name, theme_word_indices in zip(list(seed_sets.keys()), indices):
        output_dict["name"].append(seed_set_name)
        curr_seed_set = list(seed_sets[seed_set_name])
        output_dict["seed_set"].append("\t".join(sorted(curr_seed_set)))

        theme_words = [adj_embedding_df.index[word_i] for word_i in theme_word_indices]
        output_dict["theme_words"].append("\t".join(sorted(theme_words)))
        output_dict["all_words"].append(", ".join(theme_words + [item for item in curr_seed_set if item not in theme_words]))
    output_df = pd.DataFrame(output_dict)

    theme_words_output_csv = f"{output_dir}/theme_words.csv"
    theme_words_output_latex = f"{output_dir}/theme_words.tex_table"
    output_df[["name", "seed_set", "theme_words"]].to_csv(theme_words_output_csv)
    output_df[["name", "all_words"]].to_latex(theme_words_output_latex, index=False)


def main(config_path):
    # Prepare bert inputs for justifications
    justifications_df = prepare_bert_inputs(config_path)

    # prepare adj embeddings by averaging contextual embeddings
    generate_adj_embeddings(config_path, justifications_df)

    seed_sets = {
        "inclusive": {"inclusive", "exclusionary"},
        "modern": {"modern", "outdated", "traditional", "contemporary"},
        "professional": {"professional", "unprofessional"},
        "standard": {"standard", "unusual", "common", "uncommon"},
        "natural": {"natural", "fluid", "awkward", "clunky"},
    }

    build_theme_word_sets(seed_sets, config_path)


if __name__ == "__main__":
    main(f"{EXPERIMENT_PATH}/config.json")