"""
Measure the social context gender associations of the About Me sentences using a BERT masking strategy.

Author: Raymond Liu
Date: Aug 2024
"""

import ast
import torch
from transformers import BertForMaskedLM, BertTokenizer
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import json

# Define constants
MASK_INDEX = 103  # 103 corresponds to [MASK]
VARIANTS = ["man", "woman", "person"]

# Initialize the tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
model = BertForMaskedLM.from_pretrained('bert-base-uncased')
model.eval()  # Set model to evaluation mode

# Use GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)


def mask_role_noun(sentence, role_noun, start_idx, end_idx):
    """
    Mask a sentence given the role noun.
    """
    role_tokens = tokenizer.tokenize(role_noun)
    masked_sentence = f'{sentence[:start_idx]}[MASK]{sentence[end_idx:]}'
    return masked_sentence


def calculate_probabilities(sentence, variants):
    """
    Given a sentence with masked tokens,
    calculate the probabilities of certain variants in the last masked token position.
    """
    # setting add_special_tokens to False does not add the beginning of sentence/end of sentence markers (implying sentence could go on)
    tokenized_sentence = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, add_special_tokens=True)
    tokenized_sentence = tokenized_sentence.to(device)

    with torch.no_grad():
        outputs = model(**tokenized_sentence)
    logits = outputs.logits

    mask_token_indices = torch.where(tokenized_sentence.input_ids == tokenizer.mask_token_id)[1]
    last_mask_token_index = mask_token_indices[-1]
    last_mask_token_logits = logits[0, last_mask_token_index, :]
    last_mask_token_probs = torch.nn.functional.softmax(last_mask_token_logits, dim=0)
    
    probabilities = {variant: float(last_mask_token_probs[tokenizer.convert_tokens_to_ids(variant)].item()) for variant in variants}
    
    return probabilities



def calculate_bert_probabilities(input_csv, output_csv):
    """
    Calculate the social gender probabilities.
    """
    df = pd.read_csv(input_csv)
    
    results = []
    for _, row in tqdm(df.iterrows(), total=df.shape[0]):
        sentence = row['sentence']
        role_noun_info = ast.literal_eval(row['filtered_roles_data'])
        role_noun, start_idx, end_idx = role_noun_info[0]
        
        masked_sentence = mask_role_noun(sentence, role_noun, start_idx, end_idx)
        extended_sentence = masked_sentence + " I am a [MASK]"
        
        probabilities = calculate_probabilities(extended_sentence, VARIANTS)
        probabilities['role_noun'] = role_noun
        probabilities['index'] = row['index']
        probabilities['role_noun_type'] = row['role_noun_type']
        probabilities['sentence'] = extended_sentence
        results.append(probabilities)
    
    result_df = pd.DataFrame(results)
    result_df.to_csv(output_csv, index=False)


def plot_overall_graphs(bert_df, output_path):
    """
    Plot graphs displaying the overall social gender data.
    """
    masc, fem, neut = set(), set(), set()
    # role noun in our list of role nouns
    with open(f'data/role_nouns_3_way.json', 'r') as f:
        role_nouns_three_way = json.load(f)
        for role_nouns_trio in role_nouns_three_way:
            neut.add(role_nouns_trio[0])
            masc.add(role_nouns_trio[1])
            fem.add(role_nouns_trio[2])
    
    with open(f'data/role_nouns_2_way.json', 'r') as f:
        role_nouns_two_way = json.load(f)
        for role_nouns_duo in role_nouns_two_way:
            masc.add(role_nouns_duo[0])
            fem.add(role_nouns_duo[1])

    # Function to calculate normalized probabilities for two-way variants
    def normalize_two_way(df):
        df['normalized_masc'] = df['man'] / (df['man'] + df['woman'])
        return df

    # Function to calculate normalized probabilities for three-way variants
    def normalize_three_way(df):
        df['normalized_masc'] = df['man'] / (df['man'] + df['woman'])
        df['normalized_neut'] = df['person'] / (df['man'] + df['woman'] + df['person'])
        return df

    # Normalize probabilities
    data_two_way = bert_df[bert_df['role_noun_type'] == 'TWO_WAY']
    data_two_way = normalize_two_way(data_two_way)

    data_three_way = bert_df[bert_df['role_noun_type'] == 'THREE_WAY']
    data_three_way = normalize_three_way(data_three_way)

    # Boxplot for two-way variants (masculine and feminine categories)
    data_two_way_masculine = data_two_way[data_two_way['role_noun'].isin(masc)]
    data_two_way_feminine = data_two_way[data_two_way['role_noun'].isin(fem)]

    plt.figure(figsize=(10, 6))
    plt.boxplot([data_two_way_masculine['normalized_masc'], data_two_way_feminine['normalized_masc']], labels=['masc. role noun', 'fem. role noun'])
    plt.title('Normalized Probability of Variant Man (vs Woman) for Two-Way Variants')
    plt.ylabel('Normalized Probability')
    plt.savefig(f'{output_path}/2_way/_overall.png')
    plt.clf()

    # Boxplot for three-way variants comparing normalized_man_vs_woman (masculine and feminine categories)
    data_three_way_masculine = data_three_way[data_three_way['role_noun'].isin(masc)]
    data_three_way_feminine = data_three_way[data_three_way['role_noun'].isin(fem)]

    plt.figure(figsize=(10, 6))
    plt.boxplot([data_three_way_masculine['normalized_masc'], data_three_way_feminine['normalized_masc']], labels=['masc. role noun', 'fem. role noun'])
    plt.title('Normalized Probability of Variant Man (vs Woman) for Three-Way Variants')
    plt.ylabel('Normalized Probability')
    plt.savefig(f'{output_path}/3_way/_overall_gender.png')
    plt.clf()

    # Boxplot for three-way variants comparing normalized_person and normalized_man_woman (neutral and combined masculine and feminine categories)
    data_three_way_neutral = data_three_way[data_three_way['role_noun'].isin(neut)]
    data_three_way_combined_masculine_feminine = data_three_way[data_three_way['role_noun'].isin(masc.union(fem))]

    plt.figure(figsize=(10, 6))
    plt.boxplot([data_three_way_neutral['normalized_neut'], data_three_way_combined_masculine_feminine['normalized_neut']], labels=['neut. role noun', 'gendered role noun'])
    plt.title('Normalized Probability of Variant Person (vs Man/Woman) for Three-Way Variants')
    plt.ylabel('Normalized Probability')
    plt.savefig(f'{output_path}/3_way/_overall_neut.png')
    plt.clf()


def plot_role_noun_graphs(bert_df, output_path):
    """
    Plot graphs displaying the social gender for each role noun.
    """
    # role noun in our list of role nouns
    with open(f'data/role_nouns_3_way.json', 'r') as f:
        role_nouns_three_way = json.load(f)
    
    with open(f'data/role_nouns_2_way.json', 'r') as f:
        role_nouns_two_way = json.load(f)

    # Function to calculate normalized probabilities for two-way variants
    def normalize_two_way(df):
        df['normalized_masc'] = df['man'] / (df['man'] + df['woman'])
        return df

    # Function to calculate normalized probabilities for three-way variants
    def normalize_three_way(df):
        df['normalized_masc'] = df['man'] / (df['man'] + df['woman'])
        df['normalized_neut'] = df['person'] / (df['man'] + df['woman'] + df['person'])
        return df
    

    for role_noun_duo in role_nouns_two_way:
        # Normalize probabilities
        data_masc = bert_df[bert_df['role_noun'] == role_noun_duo[0]]
        data_masc = normalize_two_way(data_masc)
        data_fem = bert_df[bert_df['role_noun'] == role_noun_duo[1]]
        data_fem = normalize_two_way(data_fem)

        plt.figure(figsize=(10, 6))
        plt.boxplot([data_masc['normalized_masc'], data_fem['normalized_masc']], labels=[role_noun_duo[0], role_noun_duo[1]])
        plt.title(f'Normalized Probability of Variant Man (vs Woman) for Two-Way Variants for {role_noun_duo[0]}/{role_noun_duo[1]}')
        plt.ylabel('Normalized Probability')
        plt.savefig(f'{output_path}/2_way/{role_noun_duo[0]}.png')
        plt.clf()

    for role_noun_trio in role_nouns_three_way:
        data_neut = bert_df[bert_df['role_noun'] == role_noun_trio[0]]
        data_neut = normalize_three_way(data_neut)
        data_masc = bert_df[bert_df['role_noun'] == role_noun_trio[1]]
        data_masc = normalize_three_way(data_masc)
        data_fem = bert_df[bert_df['role_noun'] == role_noun_trio[2]]
        data_fem = normalize_three_way(data_fem)

        data_combined_masc_fem = pd.concat([data_masc, data_fem], ignore_index=True)

        plt.figure(figsize=(10, 6))
        plt.boxplot([data_masc['normalized_masc'], data_fem['normalized_masc']], labels=[role_noun_trio[1], role_noun_trio[2]])
        plt.title('Normalized Probability of Variant Man (vs Woman) for Three-Way Variants')
        plt.ylabel('Normalized Probability')
        plt.savefig(f'{output_path}/3_way/{role_noun_trio[0]}_gender.png')
        plt.clf()

        plt.figure(figsize=(10, 6))
        plt.boxplot([data_neut['normalized_neut'], data_combined_masc_fem['normalized_neut']], labels=[role_noun_trio[0], f'{role_noun_trio[1]}/{role_noun_trio[2]}'])
        plt.title('Normalized Probability of Variant Person (vs Man/Woman) for Three-Way Variants')
        plt.ylabel('Normalized Probability')
        plt.savefig(f'{output_path}/3_way/{role_noun_trio[0]}_neut.png')
        plt.clf()


def get_top_and_bottom(bert_df):
    """
    Get the sentences with the highest and lowest social gender scores.
    """
    def remove_last_occurrence(s, sub):
        pos = s.rfind(sub)
        if pos == -1:  # If the substring is not found
            return s
        return s[:pos] + s[pos+len(sub):]
    
    masc, fem, neut = set(), set(), set()
    # role noun in our list of role nouns
    with open(f'data/role_nouns_3_way.json', 'r') as f:
        role_nouns_three_way = json.load(f)
        for role_nouns_trio in role_nouns_three_way:
            neut.add(role_nouns_trio[0])
            masc.add(role_nouns_trio[1])
            fem.add(role_nouns_trio[2])

    # Function to calculate normalized probabilities for three-way variants
    def normalize_three_way(df):
        df['normalized_masc'] = df['man'] / (df['man'] + df['woman'])
        df['normalized_neut'] = df['person'] / (df['man'] + df['woman'] + df['person'])
        return df

    data_three_way = bert_df[bert_df['role_noun_type'] == 'THREE_WAY']
    data_three_way = normalize_three_way(data_three_way)

    

    pd.set_option('display.max_colwidth', None)
    print('THREE WAY TOP 20 MASC: -----')
    top_masc = data_three_way.nlargest(20, 'normalized_masc')
    sentences = top_masc['sentence'].tolist()
    orig_sentences = [remove_last_occurrence(s, "I am a [MASK]") for s in sentences]
    for s in orig_sentences:
        print(s)
    print()

    pd.set_option('display.max_colwidth', None)
    print('THREE WAY TOP 20 FEM: -----')
    top_fem = data_three_way.nsmallest(20, 'normalized_masc')
    sentences = top_fem['sentence'].tolist()
    orig_sentences = [remove_last_occurrence(s, "I am a [MASK]") for s in sentences]
    for s in orig_sentences:
        print(s)
    print()

    pd.set_option('display.max_colwidth', None)
    print('THREE WAY TOP 20 NEUT: -----')
    top_masc = data_three_way.nlargest(20, 'normalized_neut')
    sentences = top_masc['sentence'].tolist()
    orig_sentences = [remove_last_occurrence(s, "I am a [MASK]") for s in sentences]
    for s in orig_sentences:
        print(s)
    print()

    pd.set_option('display.max_colwidth', None)
    print('THREE WAY TOP 20 GENDERED: -----')
    top_fem = data_three_way.nsmallest(20, 'normalized_neut')
    sentences = top_fem['sentence'].tolist()
    orig_sentences = [remove_last_occurrence(s, "I am a [MASK]") for s in sentences]
    for s in orig_sentences:
        print(s)


# Run the main function
if __name__ == "__main__":
    about_me_data_path = " /ais/hal9000/datasets/AboutMe/about-me/individuals_pages/filtered_data.csv"
    bert_probabilities_path = "context_social_gender/bert_probabilities_data.csv"
    # calculate_bert_probabilities(about_me_data_path, bert_probabilities_path)

    output_path = 'context_social_gender'
    bert_probabilities = pd.read_csv(bert_probabilities_path)
    plot_overall_graphs(bert_probabilities, output_path)
    plot_role_noun_graphs(bert_probabilities, output_path)
    get_top_and_bottom(bert_probabilities)