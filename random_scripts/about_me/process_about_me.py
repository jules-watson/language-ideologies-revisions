"""
Download and process the about me dataset from HuggingFace.
"""

from huggingface_hub import snapshot_download, login
from blingfire import text_to_sentences

import json
import gzip
import random
import re
import pandas as pd
from collections import defaultdict

first_person_pronouns = re.compile(r'\b(I|me|my|mine)\b', re.IGNORECASE)
# third_person_pronouns = re.compile(r'\b(he|she|him|her|his|hers)\b', re.IGNORECASE)


def read_about_me(data_path):
    print(f'Reading {data_path}')

    df = pd.read_csv(data_path, index_col=0)

    print(df.head())


def get_individuals_hostnames(about_page_meta_path):
    with open(f'random_scripts/about_me/3_way_role_nouns_data.json', 'r') as f:
        role_nouns_three_way = json.load(f)
    
    with open(f'random_scripts/about_me/2_way_role_nouns_data.json', 'r') as f:
        role_nouns_two_way = json.load(f)
    
    role_nouns_set = set([role_noun for trio in role_nouns_three_way for role_noun in trio])
    role_nouns_set.update([role_noun for duo in role_nouns_two_way for role_noun in duo])

    with gzip.open(about_page_meta_path, 'rt', encoding='utf-8') as f:
        # Key: hostname, value: {key: role, value: sentence number}
        individuals_hostnames = {}
        
        for line in f:
            line_dict = json.loads(line)
            if line_dict['class'] == 'individuals':
                roles = defaultdict(list)
                for role_dict in line_dict['roles']:
                    if role_dict[3] in role_nouns_set:
                        roles[role_dict[3]].append((role_dict[0], role_dict[1], role_dict[2]))
                if roles:
                    individuals_hostnames[line_dict['hn']] = roles

                line_dict['roles']

        print("Retrieved individuals hostnames!")

        return individuals_hostnames


def save_roles_to_csv(about_pages_paths, output_dir, individuals_hns):
    """
    Unzip and save the about me dataset as CSV files, with their corresponding roles

    Note: sentence tokenizer inspired from 
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
                if line_dict['hostname'] in individuals_hns:
                    for role, role_metadata_list in individuals_hns[line_dict['hostname']].items():
                        for (sent_id, start_token, end_token) in role_metadata_list:
                            sentences = text_to_sentences(line_dict['text']).split('\n')
                            sent = sentences[sent_id].replace('\x00', '') 
                            
                            if first_person_pronouns.search(sent):
                                perspective = 'FIRST'
                            else:
                                perspective = 'THIRD'

                            data.append({
                                'file_number': i,
                                'hostname': line_dict['hostname'],
                                # 'url': line_dict['url'],
                                'sentence': sent, # line_dict['text']
                                'perspective': perspective,
                                'role': role,
                                'start_token': start_token,
                                'end_token': end_token
                            })

    df = pd.DataFrame(data)
    df.to_csv(f'{output_dir}/gendered_role_nouns.csv', index=True)


def count_first_person(gendered_rn_sents_path):
    df = pd.read_csv(gendered_rn_sents_path)
    df = df[df['perspective'] == 'FIRST']
    count = df.shape[0]
    print(f"There are {count} entries of first person role nouns.")
    count = df['sentence'].nunique()
    print(f"There are {count} unique sentences.")


def save_as_csv(about_pages_paths, output_dir, individuals_hns, random_sample=False):
    """
    Unzip and save the about me dataset as CSV files
    """    
    data = []

    # Iterate over each JSON file in the directory
    for i, file_path in enumerate(about_pages_paths):
        print(f"Reading file at path {file_path}")
        
        # Read the JSON file and load the data
        with gzip.open(file_path, 'rt', encoding='utf-8') as f:
            for line in f:
                line_dict = json.loads(line)
                if line_dict['hostname'] in individuals_hns:
                    if not random_sample or random.random() < 0.05:
                        clean_text = line_dict['text'].replace('\x00', '') 
                        data.append({
                            'hostname': line_dict['hostname'],
                            'url': line_dict['url'],
                            'text': clean_text, # line_dict['text']
                            'file_number': i
                        })
        
        if not random_sample:
            df = pd.DataFrame(data)
            output_path = f'{output_dir}/page_{i}.csv'
            df.to_csv(output_path, index=True)
            data = []

    if random_sample:
        df = pd.DataFrame(data)
        df.to_csv(f'{output_dir}/data_with_roles.csv', index=True)


def random_sentences_per_role(gendered_rn_sents_path, random_rn_sents_path, sentences_per_rn=10):
    gendered_rn_sents_df = pd.read_csv(gendered_rn_sents_path)
    gendered_rn_sents_df = gendered_rn_sents_df[gendered_rn_sents_df['perspective'] == 'FIRST']

    # Get the selected sentences
    grouped = gendered_rn_sents_df.groupby('role')
    random_sents = grouped.apply(lambda x: x.sample(n=min(sentences_per_rn, len(x)), random_state=0)).reset_index(drop=True)

    # Write the selected sentences to a new CSV file
    random_sents.to_csv(random_rn_sents_path, index=False)

    print(f"Selected sentences saved to {random_rn_sents_path}")


def download_about_me(token, download_dir):
    """
    Download the about me dataset from Huggingface:
    https://huggingface.co/datasets/allenai/aboutme
    """
    login(token=token)
    repo_id = "allenai/aboutme" 
    snapshot_download(repo_id=repo_id, repo_type="dataset", local_dir=download_dir)


if __name__=="__main__":
    # huggingface_token = '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'
    # download_dir = '/hal9000/raymliu/about-me'
    # download_about_me(huggingface_token, download_dir)

    about_zipped_dir = '/hal9000/raymliu/about-me/about_pages_zipped'
    individuals_dir = '/hal9000/raymliu/about-me/individuals_pages'

    about_page_meta_path = f'{about_zipped_dir}/about_pages_meta.json.gz'
    # individuals_hostnames = get_individuals_hostnames(about_page_meta_path)

    about_pages_paths = [f'{about_zipped_dir}/about_pages-{i}.json.gz' for i in range(14)]
    # about_pages_paths = [f'{about_zipped_dir}/about_pages-{i}.json.gz' for i in range(1)]
    # save_as_csv(about_pages_paths, individuals_dir, individuals_hostnames, random_sample=True)
    # save_roles_to_csv(about_pages_paths, individuals_dir, individuals_hostnames)

    # rn_sents_path = f'{individuals_dir}/gendered_role_nouns.csv'
    # random_rn_sents_path = f'{individuals_dir}/role_nouns_random_sentences.csv'
    # random_sentences_per_role(rn_sents_path, random_rn_sents_path, sentences_per_rn=4)

    count_first_person(f'{individuals_dir}/gendered_role_nouns.csv')

    # about_page_path = f'{individuals_dir}/page_0.csv'
    #åread_about_me(about_page_path)