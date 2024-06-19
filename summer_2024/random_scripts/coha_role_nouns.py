"""
Analyze database of COHA words.
"""

import pandas as pd
from pprint import pprint

suffixes = ["woman", "man", "person"]

def find_words_with_suffix(words_df, suffix):
    """
    Find the words in words_df that end with the requested suffix
    """
    suffix_df = words_df[words_df["word"].str.endswith(suffix)]
    return dict(zip(suffix_df["word"], suffix_df["num_sents"]))

def remove_suffix(word, suffix):
    """
    Remove the suffix from the word
    """
    if word.endswith(suffix):
        return word[:-len(suffix)]
    return word

def print_details(words_df):
    """
    Print some details about the words ending with requested suffixes
    """
    words = {} # word with requested suffix: word frequency in dictionary
    for suffix in ["man", "woman", "person", "men", "women", "people"]:
        words[suffix] = find_words_with_suffix(words_df, suffix)
        print(f"For suffix -{suffix}: there are {len(words[suffix])} words")

    base_words = {} # words with suffix removed: word frequency in dictionary
    for suffix in ["man", "woman", "person", "men", "women", "people"]:
        base_words[suffix] = {remove_suffix(word, suffix): freq for word, freq in words[suffix].items()}

    # common base words with -man, -woman, -person suffixes
    singular_common_base_words_set = set(base_words["man"].keys()) & set(base_words["woman"].keys()) & set(base_words["person"].keys())
    singular_common_words = {}
    for w in singular_common_base_words_set:
        avg_frequency = (base_words["man"][w] + base_words["woman"][w] + base_words["person"][w])/3
        singular_common_words[w + 'person'] = avg_frequency

    print(f"There are {len(singular_common_words)} words that exist with prefixes -man, -woman, and -person.")
    print(f"Some examples are:")
    sorted_data = list(sorted(singular_common_words.items(), key=lambda item: item[1], reverse=True))
    pprint(sorted_data)


def create_csv(output_file_path, words_df):
    """
    Create the CSV file with the following columns:
    word, stem (i.e. the stem of the word, with the gendered suffix removed), suffix, raw_frequency, frequency_in_a_million
    of words containing the suffix -man, -woman, and -person.

    Order this CSV by the descending frequency.
    """
    # word, stem, suffix
    gendered_words_df = words_df[words_df["word"].str.endswith(tuple(suffixes))].copy()
    gendered_words_df["suffix"] = gendered_words_df['word'].apply(lambda x: next((s for s in suffixes if x.endswith(s)), None))
    gendered_words_df['stem'] = gendered_words_df.apply(lambda row: row['word'][:-len(row['suffix'])], axis=1)

    # frequencies
    gendered_words_df.rename(columns={'num_sents': 'frequency'}, inplace=True)
    total_frequency = words_df['num_sents'].sum()
    gendered_words_df['frequency_in_a_million'] = (gendered_words_df['frequency'] / total_frequency) * 1_000_000

    output_df = gendered_words_df[['word', 'stem', 'suffix', 'frequency', 'frequency_in_a_million']]
    output_df = output_df[output_df['frequency_in_a_million'] > 0.01]
    output_df = output_df.sort_values(by='frequency', ascending=False)
    output_df.reset_index(drop=True, inplace=True)
    output_df.to_csv(output_file_path, index=True, index_label='index')



if __name__=="__main__":
    file_path = "random_scripts/gendered_nouns_data/coha_words_summary.csv"
    words_df = pd.read_csv(file_path)
    words_df["word"] = words_df["word"].fillna('')

    output_file_path = "random_scripts/gendered_nouns_data/coha_gendered_nouns.csv"
    create_csv(output_file_path, words_df)

