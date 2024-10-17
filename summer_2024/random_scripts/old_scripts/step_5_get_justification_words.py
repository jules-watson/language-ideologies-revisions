"""
Obtain word frequencies for justification words

Author: Raymond Liu
Date: Aug 2024
"""

import pandas as pd
from functools import reduce
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from common import load_json, load_csv
from constants import EXPERIMENT_PATH, MODEL_NAMES

import nltk
nltk.download('punkt')
nltk.download('stopwords')

def get_common_justification_words(revision_df):
    """
    Return a counter of words and their frequencies within a set of justifications.
    """
    def extract_content_words(sentence):
        """
        Return the set of content words within a sentence.
        """
        if pd.isna(sentence):
            return set()
        stop_words = set(stopwords.words('english'))
        words = word_tokenize(sentence.lower())
        # Remove punctuation and stop words
        content_words = {word for word in words if word.isalnum() and word not in stop_words and not word.isdigit()}
        return content_words
    
    all_words_sets = revision_df['justification'].apply(extract_content_words)
    word_justification_count = Counter()
    for word_set in all_words_sets:
        word_justification_count.update(word_set)
    
    scaled = Counter({key: value / revision_df.shape[0] for key, value in word_justification_count.items()})
    
    return scaled


def combine_common_justification_words(revision_dfs, output_path, gender):
    """
    Create a CSV of words and their frequencies within justifications, for different models.
    gender represents the gender of the pre-revision role noun, when creating the set of justifications.
    """
    relevant_revisions_dfs = []
    for model_name, revision_data in revision_dfs.items():
        if gender == 'gendered':
            relevant_revision_data = revision_data[(revision_data['variant_removed'] == True) & (revision_data['role_noun_gender'] != 'neutral')]
        elif gender == 'all':
            relevant_revision_data = revision_data[(revision_data['variant_removed'] == True)]
        else:
            relevant_revision_data = revision_data[(revision_data['variant_removed'] == True) & (revision_data['role_noun_gender'] == gender)]

        frequencies = get_common_justification_words(relevant_revision_data)
        relevant_revisions_dfs.append(pd.DataFrame(frequencies.items(), columns=['word', f'frequency_{model_name}']))
    
    combined_df = reduce(lambda left, right: pd.merge(left, right, on='word', how='outer'), relevant_revisions_dfs)
    combined_df.fillna(0, inplace=True)

    combined_df['total_frequency'] = 0.0
    for model_name in MODEL_NAMES:
        combined_df['total_frequency'] += combined_df[f'frequency_{model_name}']
    
    combined_df['average_frequency'] = combined_df[f'total_frequency'] / len(MODEL_NAMES)

     # Sort the dataframe by descending average frequency
    combined_df.sort_values(by='average_frequency', ascending=False, inplace=True)
    combined_df.to_csv(output_path, index=False)


def main(config):
    dirname = "/".join(config.split("/")[:-1])
    print(f"Analyzing: {dirname}")
    config_path = f"{dirname}/config.json"
    config = load_json(config_path)

    revision_dfs = {}
    for model_name in MODEL_NAMES:
        revision_path = f"{dirname}/{model_name}/revision.csv"
        revision_dfs[model_name] = load_csv(revision_path)
    
    for gender in ['gendered', 'neutral', 'masculine', 'feminine', 'all']:
        output_path = f'{dirname}/just_word_freqs_{gender}.csv'
        combine_common_justification_words(revision_dfs, output_path, gender)

    
if __name__ == "__main__":
    main(f"{EXPERIMENT_PATH}/config.json")