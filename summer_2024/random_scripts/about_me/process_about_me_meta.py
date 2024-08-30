import json
import csv
import gzip
import pandas as pd
from collections import defaultdict


def get_gendered_roles_count(about_page_meta_path, output_path):
    with open(f'random_scripts/about_me/3_way_role_nouns_data.json', 'r') as f:
        role_nouns_three_way = json.load(f)
    
    with open(f'random_scripts/about_me/2_way_role_nouns_data.json', 'r') as f:
        role_nouns_two_way = json.load(f)
    
    role_nouns_dict = {role_noun: 0 for trio in role_nouns_three_way for role_noun in trio}
    role_nouns_dict.update({role_noun: 0 for duo in role_nouns_two_way for role_noun in duo})

    with gzip.open(about_page_meta_path, 'rt', encoding='utf-8') as f:
        for line in f:
            line_dict = json.loads(line)
            if line_dict['class'] == 'individuals':
                for role_dict in line_dict['roles']:
                    if role_dict[3] in role_nouns_dict:
                        role_nouns_dict[role_dict[3]] += 1

    with open(output_path, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['neutral_word', 'neutral_count', 'masculine_count', 'feminine_count'])

        for trio in role_nouns_three_way:
            writer.writerow([trio[0], role_nouns_dict[trio[0]], role_nouns_dict[trio[1]], role_nouns_dict[trio[2]]])

        for duo in role_nouns_two_way:
            writer.writerow([duo[0], '', role_nouns_dict[duo[0]], role_nouns_dict[duo[1]]])



def get_roles_count(about_page_meta_path, output_path):
    roles_count = defaultdict(int)

    with gzip.open(about_page_meta_path, 'rt', encoding='utf-8') as f:
        for line in f:
            line_dict = json.loads(line)
            if line_dict['class'] == 'individuals':
                for role_dict in line_dict['roles']:
                    roles_count[role_dict[3]] += 1

    with open(output_path, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['Index', 'Role', 'Count'])

        for i, (role, count) in enumerate(sorted(roles_count.items(), key=lambda item: item[1], reverse=True)):
            writer.writerow([i, role, count])



def read_about_pages_meta(about_page_meta_path, output_path):
    data = []

    with gzip.open(about_page_meta_path, 'rt', encoding='utf-8') as f:
        for line in f:
            line_dict = json.loads(line)
            if line_dict['class'] == 'individuals':
                roles = set()
                for role in line_dict['roles']:
                    roles.add(role[3])
                data.append((line_dict['cluster'], roles))

    with open(output_path, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['Index', 'Clusters', 'Roles'])

        for i, (cluster, role_set) in enumerate(data):
            writer.writerow([i, cluster, ','.join(role_set)])


if __name__ == "__main__":
    about_zipped_dir = '/hal9000/raymliu/about-me/about_pages_zipped'
    about_page_meta_path = f'{about_zipped_dir}/about_pages_meta.json.gz'
    roles_metadata_path = 'random_scripts/about_me/roles_metadata.csv'
    roles_count_path = 'random_scripts/about_me/roles_count.csv'
    gendered_roles_count_path = 'random_scripts/about_me/gendered_roles_count.csv'

    # read_about_pages_meta(about_page_meta_path, roles_metadata_path)
    # get_roles_count(about_page_meta_path, roles_count_path)
    get_gendered_roles_count(about_page_meta_path, gendered_roles_count_path)

    