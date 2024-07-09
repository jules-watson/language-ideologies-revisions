"""
Analyze database of COHA words.
"""

import pandas as pd
from pprint import pprint

grouped_suffixes = ["person", "woman", "man"]
all_suffixes = grouped_suffixes + ["ess"]


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
    gendered_words_df = words_df[words_df["word"].str.endswith(tuple(all_suffixes))].copy()
    gendered_words_df["suffix"] = gendered_words_df['word'].apply(lambda x: next((s for s in all_suffixes if x.endswith(s)), None))
    gendered_words_df['stem'] = gendered_words_df.apply(lambda row: row['word'][:-len(row['suffix'])], axis=1)

    # frequencies
    gendered_words_df.rename(columns={'num_sents': 'frequency'}, inplace=True)
    total_frequency = words_df['num_sents'].sum()
    gendered_words_df['frequency_in_a_million'] = (gendered_words_df['frequency'] / total_frequency) * 1_000_000

    output_df = gendered_words_df[['word', 'stem', 'suffix', 'frequency', 'frequency_in_a_million']]
    output_df = output_df[output_df['frequency_in_a_million'] > 0.01]
    output_df.sort_values(by='frequency', ascending=False, inplace=True)
    output_df.reset_index(drop=True, inplace=True)
    output_df.to_csv(output_file_path, index=True, index_label='index')


def create_word_lists(output_folder_path, gendered_nouns_df):
    """
    Create 7 word lists:
    - Word stems with all 3 suffixes as words
    - Word stems with only 2/3 suffixes as words
    - Word stems with only 1/3 suffixes as words
    """
    # group by word stem
    grouped_words = gendered_nouns_df.groupby('stem').agg(
        total_frequency=('frequency', 'sum'),
        suffixes=('suffix', lambda x: set(x)),
        words=('word', list)
    ).reset_index()

    # word lists
    three_suffixes = []
    two_suffixes = {tuple(pair): [] for pair in [('man', 'woman'), ('man', 'person'), ('woman', 'person')]}
    one_suffix = {suffix: [] for suffix in grouped_suffixes}

    # Add the words to word lists!
    for _, row in grouped_words.iterrows():
        stem = row['stem']
        total_frequency = row['total_frequency']
        suffixes_set = row['suffixes']
        words = row['words']
        
        if len(suffixes_set) == 3:
            three_suffixes.append([stem, words, total_frequency])
        elif len(suffixes_set) == 2:
            for pair in two_suffixes:
                if set(pair) == suffixes_set:
                    two_suffixes[pair].append([stem, words, total_frequency])
                    break
        elif len(suffixes_set) == 1 and list(suffixes_set)[0] in one_suffix:
            suffix = list(suffixes_set)[0]
            one_suffix[suffix].append([stem, words, total_frequency])

    # Function to save data to CSV files
    def save_to_csv(data, filename):
        df_to_save = pd.DataFrame(data, columns=['stem', 'words', 'total_frequency'])
        df_to_save.sort_values(by='total_frequency', ascending=False, inplace=True)
        df_to_save.reset_index(drop=True, inplace=True)
        df_to_save.to_csv(filename, index=True, index_label='index')
    
    save_to_csv(three_suffixes, f'{output_folder_path}/three_suffixes.csv')
    for pair, data in two_suffixes.items():
        filename = f'{output_folder_path}/two_suffixes_{"_".join(pair)}.csv'
        save_to_csv(data, filename)
    for suffix, data in one_suffix.items():
        filename = f'{output_folder_path}/one_suffix_{suffix}.csv'
        save_to_csv(data, filename)


def create_word_list_ess(output_folder_path, gendered_nouns_df):
    """
    Created word list of words that end with "ess"
    """
    df_ess = gendered_nouns_df[gendered_nouns_df['word'].str.contains(r'(?<!l|n)ess$')]
    df_ess_filtered = df_ess[['stem', 'word', 'frequency']]

    df_ess_filtered.to_csv(f'{output_folder_path}/suffix_ess.csv', index=False)


if __name__=="__main__":
    raw_data_path = "random_scripts/gendered_nouns_data/coha_words_summary.csv"
    gendered_nouns_path = "random_scripts/gendered_nouns_data/coha_gendered_nouns.csv"
    output_folder_path = "random_scripts/gendered_nouns_data"

    # words_df = pd.read_csv(raw_data_path)
    # words_df["word"] = words_df["word"].fillna('')
    # create_csv(gendered_nouns_path, words_df)

    # gendered_nouns_df = pd.read_csv(gendered_nouns_path)
    # create_word_lists(output_folder_path, gendered_nouns_df)
    # create_word_list_ess(output_folder_path, gendered_nouns_df)
