import collections
import numpy as np
import os
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import spacy
from transformers import BertTokenizer, BertModel
import tqdm
import torch


CONFIG_PATHS = [
        "analyses/piloting_jan1/improve/config.json",
        "analyses/piloting_jan1/improve_if_needed/config.json",
        "analyses/piloting_jan1/revise/config.json",
        "analyses/piloting_jan1/revise_if_needed/config.json"
]

OUTPUT_DIR = "random_scripts/mar1_analyses"


nlp = spacy.load("en_core_web_sm")

# Load pre-trained model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')


def extract_adjectives_from_justifications(justifications_df):
    result = collections.Counter()
    for _, row in justifications_df.iterrows():
        justification = row["sbert_strings"]
        adjectives = [tok.text.lower() for tok in nlp(justification) if tok.pos_ == "ADJ"]
        result.update(collections.Counter(adjectives))
    return result


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


def get_contextual_embeddings(justifications_df, adjectives):
    result = collections.defaultdict(list)
    for _, row in tqdm.tqdm(justifications_df.iterrows()):
        justification = row["sbert_strings"]

        inputs = tokenizer(justification, return_tensors='pt')
        with torch.no_grad():
            outputs = model(**inputs)

        last_hidden_states = outputs.last_hidden_state
        tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

        cumulative_token = ""
        cumulative_embedding = []
        for token, embedding in zip(tokens, last_hidden_states[0]):
            # print(f"Token: {token}")

            if token.startswith("##"): #  This is a continuation
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


def generate_adj_embeddings():

    # Step 1: Load justifications
    justifications_df = load_justifications(CONFIG_PATHS)

    # Step 2: get adjectives
    adjectives = extract_adjectives_from_justifications(justifications_df)

    # Step 3: generate contextual embeddings for each adjective
    # occurrence
    contextual_embeddings_output_path = f"{OUTPUT_DIR}/contextual_embeddings.csv"
    if os.path.exists(contextual_embeddings_output_path):
        contextual_embeddings = pd.read_csv(contextual_embeddings_output_path)
        contextual_embeddings["embedding"] = [eval(item) for item in contextual_embeddings["embedding"]]
    else:
        contextual_embeddings = get_contextual_embeddings(justifications_df, adjectives)
        contextual_embeddings.to_csv(f"{OUTPUT_DIR}/contextual_embeddings.csv")

    # Step 4: average embeddings per adjective and store them
    adj_embedding_output_path = f"{OUTPUT_DIR}/adj_embddings.csv"
    adj_embedding_dict = collections.defaultdict(list)
    for adj, adj_embeddings in contextual_embeddings.groupby("adjective"):
        mean_embedding = np.array(list(adj_embeddings["embedding"])).mean(axis=0)
        adj_embedding_dict["adjective"].append(adj)
        adj_embedding_dict["embedding"].append(list(mean_embedding))
    adj_embedding_df = pd.DataFrame(adj_embedding_dict)
    adj_embedding_df.to_csv(adj_embedding_output_path)


def build_theme_word_sets(seed_sets, n=15):

    # Step 1: load embeddings
    adj_embedding_path = f"{OUTPUT_DIR}/adj_embddings.csv"
    adj_embedding_df = pd.read_csv(adj_embedding_path)
    adj_embedding_df["embedding"] = [eval(item) for item in adj_embedding_df["embedding"]]
    adj_embedding_df = adj_embedding_df.set_index("adjective")

    # Step 3: Construct embeddings for each seed set
    seed_embeddings = []
    for seed_set in seed_sets:
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
    for seed_set, theme_word_indices in zip(seed_sets, indices):
        output_dict["seed_set"].append("\t".join(seed_set))

        theme_words = [adj_embedding_df.index[word_i] for word_i in theme_word_indices]
        output_dict["theme_words"].append("\t".join(theme_words))
    output_df = pd.DataFrame(output_dict)

    theme_words_output_path = f"{OUTPUT_DIR}/theme_words.csv"
    output_df.to_csv(theme_words_output_path)




if __name__ == "__main__":
    generate_adj_embeddings()

    seed_sets = [
        {"inclusive", "exclusionary"},
        {"modern", "outdated", "traditional", "contemporary"},
        {"professional"},
        {"standard", "common"},
        {"natural", "fluid", "awkward", "clunky"},
        {"authentic"},
        {"sexist"}
    ]

    # build_theme_word_sets(seed_sets)