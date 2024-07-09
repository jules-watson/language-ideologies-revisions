"""
Download and process the about me dataset from HuggingFace.
"""

from huggingface_hub import snapshot_download, login
from blingfire import text_to_sentences

import ast
import json
import gzip
import re
import pandas as pd
import spacy
from collections import defaultdict

nlp = spacy.load('en_core_web_sm')

# TODO: this should be revised!
gendered_words = set([
    'man', 'woman', 'men', 'women', 'boy', 'girl', 'gentleman', 'lady', 'guys', 'gals', 'lads', 'lasses',
    'father', 'mother', 'dad', 'mom', 'daddy', 'mommy', 'papa', 'mama',
    'stepfather', 'stepmother', 'stepdad', 'stepmom',
    'son', 'daughter', 'brother', 'sister', 'bro', 'sis',
    'stepson', 'stepdaughter', 'brother-in-law', 'sister-in-law',
    'uncle', 'aunt', 'nephew', 'niece',
    'grandfather', 'grandmother', 'grandpa', 'grandma', 'granddad', 'grandmom',
    'king', 'queen',
    'husband', 'wife', 'boyfriend', 'girlfriend', 'fiancé', 'fiancée',
    'sir', 'madam', 'miss', 'mr', 'mrs', 'ms',
])

first_person_pronouns = re.compile(r'\b(I|me|my|mine|myself|we|us|our|ours|ourselves)\b', re.IGNORECASE)
third_person_pronouns = re.compile(r'\b(he|she|him|her|his|hers|himself|herself)\b', re.IGNORECASE)

def read_about_me(data_path):
    print(f'Reading {data_path}')

    df = pd.read_csv(data_path, index_col=0)

    print(df.head())


def get_individuals_hostnames(about_page_meta_path):
    """
    Return a data structure representing the hostnames that correspond to individuals in the dataset.
    """
    with gzip.open(about_page_meta_path, 'rt', encoding='utf-8') as f:
        # Key: hostname, 
        # value: {key: sentence number, 
        #         value: list of tuples (role, start token, end token)
        # }
        individuals_hostnames = {}
        
        for line in f:
            line_dict = json.loads(line)
            if line_dict['class'] == 'individuals':
                roles = defaultdict(list)
                for role_dict in line_dict['roles']:
                    roles[role_dict[0]].append((role_dict[3], role_dict[1], role_dict[2]))
                if roles:
                    individuals_hostnames[line_dict['hn']] = roles

        print("Retrieved individuals hostnames!")

        return individuals_hostnames


def save_data_to_csv(about_pages_paths, output_dir, individuals_hns):
    """
    Unzip and save the about me dataset as CSV files, with their corresponding roles

    Note: sentence tokenizer taken from 
    https://github.com/lucy3/whos_filtered/blob/main/code/identity_measures/personas/get_role_occurrences.py
    """    
    data = []

    # Iterate over each JSON file in the directory
    for i, file_path in enumerate(about_pages_paths):
        print(f"Reading file at path {file_path}")
        
        # Read the JSON file and load the data
        with gzip.open(file_path, 'rt', encoding='utf-8') as f:
            for line in f:
                line_dict = json.loads(line)
                if line_dict['hostname'] in individuals_hns: # this about me page belongs to an individual
                    sentences = text_to_sentences(line_dict['text']).split('\n')

                    # Iterate through sentences representing "I am role" statements
                    for sent_id, sent_metadata_list in individuals_hns[line_dict['hostname']].items():
                        sent = sentences[sent_id].replace('\x00', '') 

                        third_person_count = len(third_person_pronouns.findall(sent))
                        first_person_count = len(first_person_pronouns.findall(sent))
                        
                        if third_person_count > first_person_count:
                            perspective = 'MORE_THIRD'
                        elif third_person_count > 0 and third_person_count < first_person_count:
                            perspective = 'LESS_THIRD'
                        elif first_person_count > 0 and third_person_count == 0:
                            perspective = 'COMPLETE_FIRST'
                        else:
                            perspective = 'NONE'

                        data.append({
                            'file_number': i,
                            'hostname': line_dict['hostname'],
                            # 'url': line_dict['url'],
                            'sentence': sent, # line_dict['text']
                            'perspective': perspective,
                            'roles_data': sent_metadata_list
                        })

    df = pd.DataFrame(data)
    df.to_csv(f'{output_dir}/data_with_roles.csv', index=True)


def filter_csv(data_path, output_dir):
    """
    Filter the data with roles, based on various criteria.
    """
    df = pd.read_csv(data_path)

    print(f'We start with {df.shape[0]} rows.')

    # role noun in our list of role nouns
    with open(f'data/role_nouns_3_way.json', 'r') as f:
        role_nouns_three_way = json.load(f)
    
    with open(f'data/role_nouns_2_way.json', 'r') as f:
        role_nouns_two_way = json.load(f)

    role_nouns_three_way_set = set([role_noun for trio in role_nouns_three_way for role_noun in trio])
    role_nouns_two_way_set = set([role_noun for trio in role_nouns_two_way for role_noun in trio])
    role_nouns_set = role_nouns_three_way_set.union(role_nouns_two_way_set)

    def filter_and_retain_roles(roles_metadata_str):
        """
        For a string representation of roles in a sentence:
        filter and retain only the tuples with roles in role_nouns_set.
        """
        roles_metadata_list = ast.literal_eval(roles_metadata_str)
        filtered_roles = [role_tuple for role_tuple in roles_metadata_list if role_tuple[0] in role_nouns_set]
        return filtered_roles
        
    # Apply the function to filter and retain roles, and filter out rows with an empty list
    df['filtered_roles_data'] = df['roles_data'].apply(filter_and_retain_roles)
    filtered_df = df[df['filtered_roles_data'].apply(bool)]
    print(f'After filtering for the specific role nouns: we have {filtered_df.shape[0]} rows.')

    # first person
    print(f'A perspective count here:') 
    for p in ['MORE_THIRD', 'LESS_THIRD', 'COMPLETE_FIRST', 'NONE']:
        print(f'   there are {filtered_df[filtered_df["perspective"] == p].shape[0]} rows with perspective {p}')
    filtered_df = filtered_df[filtered_df['perspective'] == 'COMPLETE_FIRST']
    print(f'After filtering for first person perspective: we have {filtered_df.shape[0]} rows.')

    # length limit (500 characters)
    filtered_df = filtered_df[filtered_df['sentence'].str.len() <= 500]
    print(f'After filtering for length (max 500 chars): we have {filtered_df.shape[0]} rows.')

    def label_role_nouns(row):
        """
        Label each sentence based on the type of role nouns they have
        """
        roles_metadata = row['filtered_roles_data']
        if len(roles_metadata) > 1:
            return "MORE_THAN_ONE"
        elif roles_metadata[0][0] in role_nouns_three_way_set:
            return "THREE_WAY"
        else:
            assert roles_metadata[0][0] in role_nouns_two_way_set
            return "TWO_WAY"
    
    filtered_df['role_noun_type'] = filtered_df.apply(label_role_nouns, axis=1)
    filtered_df.drop(filtered_df[filtered_df['role_noun_type'] == 'MORE_THAN_ONE'].index, inplace=True)
    print(f'After filtering for max 1 gendered role noun: we have {filtered_df.shape[0]} rows.')
    print(f"   Within those rows: {filtered_df[filtered_df['role_noun_type'] == 'THREE_WAY'].shape[0]} three way rows and {filtered_df[filtered_df['role_noun_type'] == 'TWO_WAY'].shape[0]} two way rows.")

    # remove proper nouns and gendered terms
    gendered_words.update(role_nouns_set)

    def filter_sentences(row): 
        """
        Return True if a sentence has no pronouns, and no gendered words (or just 1 gendered word that is a role noun)
        """
        doc = nlp(row['sentence'])
        tokens = {token.text.lower() for token in doc}
        proper_nouns = [token for token in doc if token.pos_ == 'PROPN']
        
        # Check for proper nouns
        if proper_nouns:
            return False
        
        # Check for gendered words
        gendered_words_in_sentence = tokens.intersection(gendered_words)
        
        if len(gendered_words_in_sentence) == 0:
            return True
        elif len(gendered_words_in_sentence) == 1:
            return gendered_words_in_sentence.pop() in role_nouns_set 
        else:
            return False

    filtered_df['keep'] = filtered_df.apply(filter_sentences, axis=1)
    filtered_df = filtered_df[filtered_df['keep']].drop(columns=['keep'])

    print(f'After filtering for proper nouns and gendered terms: we have {filtered_df.shape[0]} rows.')
    
    # Write final filtered data to file
    filtered_df.reset_index(drop=True, inplace=True)
    filtered_df.to_csv(f'{output_dir}/filtered_data.csv', index=True, index_label='index')


def download_about_me(token, download_dir):
    """
    Download the about me dataset from Huggingface:
    https://huggingface.co/datasets/allenai/aboutme
    """
    login(token=token)
    repo_id = "allenai/aboutme" 
    snapshot_download(repo_id=repo_id, repo_type="dataset", local_dir=download_dir)


if __name__=="__main__":
    # Download the raw About Me dataset
    # huggingface_token = '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'
    download_dir = '/hal9000/raymliu/about-me'
    # download_about_me(huggingface_token, download_dir)

    about_zipped_dir = f'{download_dir}/about_pages_zipped'
    individuals_dir = f'{download_dir}/individuals_pages'
    about_page_meta_path = f'{about_zipped_dir}/about_pages_meta.json.gz'

    # Retrieve the hostnames corresponding to individuals
    individuals_hostnames = get_individuals_hostnames(about_page_meta_path)

    # Unzip and save the data as a CSV file
    about_pages_paths = [f'{about_zipped_dir}/about_pages-{i}.json.gz' for i in range(14)]
    save_data_to_csv(about_pages_paths, individuals_dir, individuals_hostnames)

    # Filter the CSV file for relevant rows
    filter_csv(f'{individuals_dir}/data_with_roles.csv', individuals_dir)