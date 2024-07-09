"""
Calculate some statistics about the About Me dataset
"""

import pandas as pd
import csv
import json
import sys
import re
from tqdm import trange
from collections import defaultdict

csv.field_size_limit(sys.maxsize)


def pronouns(about_me_data, output_dir):
    print("Processing pronouns")
    output_csv_path = f'{output_dir}/results_pronouns.csv'

    with open('random_scripts/about_me/pronouns.json', 'r') as f:
        pronouns_data = json.load(f)

    pronoun_patterns = {
        'neutral': [re.compile(r'\b' + re.escape(pronoun) + r'\b', re.IGNORECASE) for pronoun in pronouns_data['neutral']],
        'masculine': [re.compile(r'\b' + re.escape(pronoun) + r'\b', re.IGNORECASE) for pronoun in pronouns_data['masculine']],
        'feminine': [re.compile(r'\b' + re.escape(pronoun) + r'\b', re.IGNORECASE) for pronoun in pronouns_data['feminine']],
    }

    counts = {
        'neutral': {pronoun: 0 for pronoun in pronouns_data['neutral']},
        'masculine': {pronoun: 0 for pronoun in pronouns_data['masculine']},
        'feminine': {pronoun: 0 for pronoun in pronouns_data['feminine']},
    }

    for i in trange(len(about_me_data)):
        line = about_me_data[i]
        for gender, patterns in pronoun_patterns.items():
            for pattern, pronoun in zip(patterns, pronouns_data[gender]):
                if pattern.search(line):
                    counts[gender][pronoun] += 1
    
    # Calculate the total sum for each gender
    total_sums = [sum(counts[gender].values()) for gender in ['neutral', 'masculine', 'feminine']]

    # Write the results to a new CSV file
    with open(output_csv_path, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['pronoun', 'neutral_count', 'masculine_count', 'feminine_count'])
        
        for gender in ['neutral', 'masculine', 'feminine']:
            for pronoun in pronouns_data[gender]:
                row = [pronoun]
                if gender == 'neutral':
                    row.extend([counts['neutral'][pronoun], '', ''])
                elif gender == 'masculine':
                    row.extend(['', counts['masculine'][pronoun], ''])
                elif gender == 'feminine':
                    row.extend(['', '', counts['feminine'][pronoun]])
                writer.writerow(row)
        
        writer.writerow(['TOTAL SUM', total_sums[0], total_sums[1], total_sums[2]])


def kinship_terms(about_me_data, output_dir):
    print("Processing kinship terms")
    output_csv_path = f'{output_dir}/results_kinship_terms.csv'

    # Get the kinship terms
    kinship_df = pd.read_csv('random_scripts/about_me/kinship_terms.csv')
    kinship_patterns = defaultdict(lambda: {'neut': {}, 'masc': {}, 'fem': {}})
    for _, row in kinship_df.iterrows():
        term = row['lemma'] # enforce singular versions of pronouns
        group = row['group']
        if row['gender_neutral']:
            kinship_patterns[group]['neut'][term] = re.compile(r'\b' + re.escape(term) + r'\b', re.IGNORECASE)
        elif row['masculine']:
            kinship_patterns[group]['masc'][term] = re.compile(r'\b' + re.escape(term) + r'\b', re.IGNORECASE)
        else:
            kinship_patterns[group]['fem'][term] = re.compile(r'\b' + re.escape(term) + r'\b', re.IGNORECASE)

    counts = defaultdict(lambda: {'neut': defaultdict(int), 'masc': defaultdict(int), 'fem': defaultdict(int)})

    for i in trange(len(about_me_data)):
        line = about_me_data[i]
        for kinship_group in kinship_patterns:
            for gender, patterns_dict in kinship_patterns[kinship_group].items():
                for term, pattern in patterns_dict.items():
                    if pattern.search(line):
                        counts[kinship_group][gender][term] += 1

    total_sums = [0, 0, 0]
    
    with open(output_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['kinship_group', 'term', 'neutral_count', 'masculine_count', 'feminine_count'])
        
        for kinship_group in kinship_patterns:
            for gender in kinship_patterns[kinship_group]:
                for term in kinship_patterns[kinship_group][gender]:
                    row = [kinship_group, term]
                    if gender == 'neut':
                        row.extend([counts[kinship_group]['neut'][term], '', ''])
                    elif gender == 'masc':
                        row.extend(['', counts[kinship_group]['masc'][term], ''])
                    elif gender == 'fem':
                        row.extend(['', '', counts[kinship_group]['fem'][term]])
                    writer.writerow(row)
            
            sums = [sum(counts[kinship_group][gender].values()) for gender in ['neut', 'masc', 'fem']]
            total_sums = [a + b for a, b in zip(total_sums, sums)]
            writer.writerow([f'TOTAL SUM FOR GROUP', kinship_group, sums[0], sums[1], sums[2]])
        
        writer.writerow([f'TOTAL SUM', '', total_sums[0], total_sums[1], total_sums[2]])



def role_nouns(about_me_data, output_dir, is_test=False):
    print("Processing role nouns")
    if is_test:
        output_csv_path = f'{output_dir}/results_role_nouns_test.csv'
    else:
        output_csv_path = f'{output_dir}/results_role_nouns.csv'

    with open(f'{output_dir}/3_way_role_nouns_data.json', 'r') as f:
        role_nouns_three_way = json.load(f)

    with open(f'{output_dir}/2_way_role_nouns_data.json', 'r') as f:
        role_nouns_two_way = json.load(f)

    role_noun_patterns = {
        noun[0]: {
            'neut': re.compile(r'\b{}\b'.format(re.escape(noun[0])), re.IGNORECASE),
            'masc': re.compile(r'\b{}\b'.format(re.escape(noun[1])), re.IGNORECASE),
            'fem': re.compile(r'\b{}\b'.format(re.escape(noun[2])), re.IGNORECASE)
        } for noun in role_nouns_three_way
    }
    role_noun_patterns.update({
        noun[0]: {
            'masc': re.compile(r'\b{}\b'.format(re.escape(noun[0])), re.IGNORECASE),
            'fem': re.compile(r'\b{}\b'.format(re.escape(noun[1])), re.IGNORECASE),
        } for noun in role_nouns_two_way
    })

    counts = defaultdict(lambda: defaultdict(lambda: 0, {'masc': 0, 'fem': 0}))

    for i in trange(len(about_me_data)):
        text = about_me_data[i]
        for role_noun in role_noun_patterns:
            for gender, pattern in role_noun_patterns[role_noun].items():
                if pattern.search(text):
                    counts[role_noun][gender] += 1

    total_sums = [0, 0, 0]
    with open(output_csv_path, 'w') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['Neutral', 'Neutral Count', 'Masculine Count', 'Feminine Count'])

        for role_noun in role_noun_patterns:
            role_noun_counts = [counts[role_noun][gender] for gender in ['neut', 'masc', 'fem']]
            role_noun_counts = role_noun_counts if len(role_noun_counts) == 3 else [''] + role_noun_counts
            writer.writerow([role_noun] + role_noun_counts)
            total_sums = [a + (b if isinstance(b, int) else 0) for a, b in zip(total_sums, role_noun_counts)]
        
        writer.writerow(['TOTAL SUM', total_sums[0], total_sums[1], total_sums[2]])
    

def read_about_me(about_page_path):
    """
    Read the about me CSV file
    """
    with open(about_page_path, 'r') as f:
        reader = csv.DictReader(f)
        lines = [row['text'] for row in reader]
    
    print("Finished reading data")
    return lines


if __name__=="__main__":
    individuals_dir = '/hal9000/raymliu/about-me/individuals_pages'
    about_page_path = f'{individuals_dir}/random_sample.csv'
    # about_page_path = f'{individuals_dir}/random_sample_small.csv'
    output_dir = 'random_scripts/about_me'

    about_me_data = read_about_me(about_page_path)

    role_nouns(about_me_data, output_dir)
    # kinship_terms(about_me_data, output_dir)
    # pronouns(about_me_data, output_dir)