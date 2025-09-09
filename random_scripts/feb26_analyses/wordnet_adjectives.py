import collections
import spacy
from nltk.corpus import wordnet as wn
import numpy as np
import pandas as pd
import networkx as nx


CONFIG_PATHS = [
        "analyses/piloting_jan1/improve/config.json",
        "analyses/piloting_jan1/improve_if_needed/config.json",
        "analyses/piloting_jan1/revise/config.json",
        "analyses/piloting_jan1/revise_if_needed/config.json"
]

OUTPUT_DIR = "random_scripts/feb26_analyses"


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


def extract_adjectives_from_justifications(justifications_df):
    result = collections.Counter()
    for _, row in justifications_df.iterrows():
        justification = row["sbert_strings"]
        adjectives = [tok.text.lower() for tok in nlp(justification) if tok.pos_ == "ADJ"]
        result.update(collections.Counter(adjectives))
    return result


def group_adjectives():
    justifications_df = load_justifications(CONFIG_PATHS)
    adjectives = extract_adjectives_from_justifications(justifications_df)

    # TODO - group adjectives based on wordnet
    g = nx.Graph()
    g.add_nodes_from(list(adjectives))
    for adj in adjectives:
        for synset in wn.synsets(adj):
            for lemma in synset.lemmas():
                
                # add edge from adj to this lemma, if they have different names
                lemma_text = lemma.name()
                if lemma_text != adj and lemma_text in g:
                    g.add_edge(adj, lemma_text)

                # add edge from this lemma to its antonyms
                for antonym in lemma.antonyms():
                    antonym_text = antonym.name()
                    if antonym_text != lemma_text and antonym_text in g:
                        g.add_edge(lemma_text, antonym_text)
            
            for sim_synset in synset.similar_tos():

                for lemma in sim_synset.lemmas():

                    # add edge from adj to this lemma, if they have different names
                    lemma_text = lemma.name()
                    if lemma_text != adj and lemma_text in g:
                        g.add_edge(adj, lemma_text)

                    # add edge from this lemma to its antonyms
                    for antonym in lemma.antonyms():
                        antonym_text = antonym.name()
                        if antonym_text != lemma_text and antonym_text in g:
                            g.add_edge(lemma_text, antonym_text)

                        








if __name__ == "__main__":
    group_adjectives()