"""
Download and process the about me dataset from HuggingFace.
"""

#from huggingface_hub import snapshot_download, login
from blingfire import text_to_sentences

import ast
from collections import defaultdict
import gzip
import json
import matplotlib.pyplot as plt
import pandas as pd
import re
import spacy
import seaborn as sns


nlp = spacy.load('en_core_web_sm')

ROLE_NOUNS_PATH = "data/role_nouns_expanded.csv"
NAMES_PATH = "data/1998.txt"    # 1998 US Social Security names dataset

first_person_pronouns = re.compile(r'\b(I|me|my|mine|myself|we|us|our|ours|ourselves)\b', re.IGNORECASE)
third_person_pronouns = re.compile(r'\b(he|she|him|her|his|hers|himself|herself)\b', re.IGNORECASE)


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
    
    Select only pages about individuals (i.e., with hostnames in individuals_hns)

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

                        data.append({
                            'file_number': i,
                            'hostname': line_dict['hostname'],
                            'sentence': sent,
                            'roles_data': sent_metadata_list
                        })

    df = pd.DataFrame(data)
    df.to_csv(f'{output_dir}/data_with_roles.csv', index=True)


def load_role_nouns_sets(role_nouns_path=ROLE_NOUNS_PATH):
    role_nouns_df = pd.read_csv(role_nouns_path)
    role_nouns = list(role_nouns_df["neutral"]) + list(role_nouns_df["feminine"]) + list(role_nouns_df["masculine"])
    return set(role_nouns)


def load_role_nouns_lookup():
    role_nouns_df = pd.read_csv(ROLE_NOUNS_PATH)
    role_nouns_df["role_noun_set"] = role_nouns_df["neutral"]
    role_nouns_lookup = role_nouns_df.melt(value_vars=["neutral", "masculine", "feminine"], id_vars=["role_noun_set"])
    role_nouns_lookup = role_nouns_lookup.rename(columns={"variable": "gender", "value": "variant"})
    role_nouns_lookup = role_nouns_lookup.set_index("variant")
    return role_nouns_lookup


def  load_gendered_words():
    # SINGULAR FORMS ONLY - we use lemmas when checking for gendered words
    gendered_words = set([
        # WORDS ABOUT GENDER and SEX
        'man', 'woman',  
        'male', 'female',
        'boy', 'girl',  
        'gentleman', 'lady', 
        'guy', 'gal', 
        'lad', 'lass', 
        'dude',
        'transman', 'transwoman', 'transmasculine', 'transfeminine', 'transmasc', 'transfemme',
        'nonbinary', 'genderqueer', 
        'lesbian',

        # KINSHIP TERMS - Kemp et al (2012); Watson et al. (2023 CogSci)
        'father', 'mother', 'dad', 'mom', 'mum', 'daddy', 'mommy', 'papa', 'mama',
        'stepfather', 'stepmother', 'stepdad', 'stepmom',
        'son', 'daughter',
        'brother', 'sister', 'bro', 'sis',
        'stepson', 'stepdaughter',
        'grandson', 'granddaughter',
        'uncle', 'aunt', 'auntie',
        'nephew', 'niece',
        'grandfather', 'grandmother', 'grandpa', 'grandma', 'granddad', 'grandmom',
        'husband', 'wife', 'hubby',
        'boyfriend', 'girlfriend', 'bf', 'gf',
        'fiancé', 'fiancée', 'fiance', 'fiancee',

        # ROYALTY / FANTASY
        'king', 'queen', 'prince', 'princess',
        'priest', 'priestess',
        'superman', 'superwoman',

        # ADDRESS TERMS
        'sir', 'madam', 'miss', 'mr', 'mrs', 'ms',
    ])

    # words with gendered affixes from Bartl & Leavy
    bartl_leavy_df = pd.read_csv("data/bartl_leavy_replacements.csv")
    bartl_leavy_df["has_variant"] = (bartl_leavy_df["word"] !=  bartl_leavy_df["variant"])
    gendered_words.update([item.lower() for item in bartl_leavy_df["word"]])
    gendered_words.update([item.lower() for item in bartl_leavy_df[bartl_leavy_df["has_variant"]]["variant"]])

    # Papineau role nouns (2-way and 3-way)
    with open("data/papineau_role_nouns_full.json", "r") as f:
        papineau_role_nouns = json.load(f)
    for rn_set in papineau_role_nouns:
        gendered_words.update(rn_set)

    return gendered_words


def load_names():
    df = pd.read_csv(NAMES_PATH, names=["name", "sex", "count"])
    return set(df["name"])


def filter_role_nouns(df, role_nouns_set):

    def filter_and_retain_roles(roles_metadata_str):
        """
        For a string representation of roles in a sentence:
        filter and retain only the tuples with roles in role_nouns_set.
        """
        roles_metadata_list = ast.literal_eval(roles_metadata_str)
        filtered_roles = [role_tuple for role_tuple in roles_metadata_list if role_tuple[0] in role_nouns_set]
        return filtered_roles
        
    # Apply the function to filter and retain roles, and select rows with exactly 1 role noun
    df['filtered_roles_data'] = df['roles_data'].apply(filter_and_retain_roles)
    df["n_roles"] = df["filtered_roles_data"].apply(len)
    filtered_df = df[df['n_roles'] == 1]
    
    print(f'After filtering for the specific role nouns: we have {filtered_df.shape[0]} rows.')
    return filtered_df


def assess_perspective(sent):
    third_person_count = len(third_person_pronouns.findall(sent))
    first_person_count = len(first_person_pronouns.findall(sent))

    if first_person_count > 0 and third_person_count == 0:
        return "COMPLETE_FIRST"
    return "OTHER"


def filter_first_person(filtered_df):
    filtered_df.loc[:, "perspective"] = filtered_df["sentence"].apply(assess_perspective)
    filtered_df = filtered_df[filtered_df['perspective'] == 'COMPLETE_FIRST']
    print(f'After filtering for first person perspective: we have {filtered_df.shape[0]} rows.')
    return filtered_df


def contains_quotes(sent):
    if "\"" in sent or "“" in sent or "”" in sent:
        return 1
    return 0
    

def filter_remove_quotes(filtered_df):
    filtered_df.loc[:, "contains_quotes"] = filtered_df["sentence"].apply(contains_quotes)
    filtered_df = filtered_df[filtered_df["contains_quotes"] == 0]
    print(f'After filtering to remove quotes: we have {filtered_df.shape[0]} rows.')
    return filtered_df


def filter_gendered_words(filtered_df, role_nouns_set, gendered_words, names):

    def filter_sentences(row): 
        """
        Return True if a sentence has no gendered pronouns, 
        and no gendered words (or just 1 gendered word that is a role noun)
        """
        doc = row["tokens"]
        lemmas = [token.lemma_.lower() for token in doc]
        
        # Check NER for mentions of people
        for ent in doc.ents:
            if ent.label_ ==  "PERSON":
                return False
        
        # Exclude sentences containing the word "name" 
        # (want to exclude cases like  "My name is ...")
        if "name" in lemmas:
            return False
        
        # Exclude sentences containing names from the names list - only considers
        # them to be a match if capitalization matches
        names_in_sentence = {token.text for token in doc if token.text in names}
        if len(names_in_sentence) > 0:
            return False
        
        # Check for gendered words
        gendered_words_in_sentence = [word for word in lemmas if word in gendered_words]
        if (len(gendered_words_in_sentence) == 1 and 
            gendered_words_in_sentence[0] not in role_nouns_set):
            return False
        elif len(gendered_words_in_sentence) > 1:
            return False
        
        # Check for multiple role nouns
        role_nouns_re = re.compile(r'\b(' + "|".join(role_nouns_set) +  r')\b', re.IGNORECASE)
        role_nouns_in_sentence = role_nouns_re.findall(row["sentence"])
        if len(role_nouns_in_sentence) > 1:
            return False
    
        return True

    filtered_df['keep'] = filtered_df.apply(filter_sentences, axis=1)
    filtered_df = filtered_df[filtered_df['keep']].drop(columns=['keep'])

    print(f'After filtering for proper nouns and gendered terms: we have {filtered_df.shape[0]} rows.')
    return filtered_df


def filter_csv(data_path, output_dir):
    """
    Filter the data with roles, based on various criteria.
    """
    df = pd.read_csv(data_path)
    print(f'We start with {df.shape[0]} rows.')  #  5721074

    # Load role nouns and gendered words
    role_nouns_set = load_role_nouns_sets()
    gendered_words = load_gendered_words()
    names = load_names()

    # Filter to select only sentences that contain exactly one role noun from our role noun set
    filtered_df = filter_role_nouns(df, role_nouns_set)

    # first person
    filtered_df = filter_first_person(filtered_df)

    # remove sentences containing quotation marks
    filtered_df = filter_remove_quotes(filtered_df)

    # add column to filtered_df with spacy output (so we can use it for next steps)
    filtered_df.loc[:, "tokens"] = filtered_df["sentence"].apply(nlp)

    # remove proper nouns and gendered terms
    filtered_df = filter_gendered_words(filtered_df, role_nouns_set, gendered_words, names)

    # remove duplicate sentences
    filtered_df = filtered_df.drop_duplicates(subset=['sentence'], keep='first')
    print(f'After filtering to remove duplicate sentences: we have {filtered_df.shape[0]} rows.')

    # make a histogram of sentence lengths
    filtered_df["sentence_length"] = filtered_df["tokens"].apply(len)
    sns.histplot(data=filtered_df, x="sentence_length")
    plt.savefig("random_scripts/Nov25/about_me_filtering_sentence_lengths.png")
    plt.xlim(0, 200)

    # length limit (20 words)
    # mean: 22.70511312855732
    # median: 20
    # sd: 12.418984527567865
    sent_min = filtered_df['sentence_length'].mean() - filtered_df['sentence_length'].std()  # 10.3
    sent_max = filtered_df['sentence_length'].mean() + filtered_df['sentence_length'].std()  # 35.1
    filtered_df = filtered_df[(filtered_df['sentence_length'] <= sent_max) & (filtered_df['sentence_length'] >= sent_min)]
    print(f'After filtering for length (11-35 words): we have {filtered_df.shape[0]} rows.')

    # Write final filtered data to file
    filtered_df.reset_index(drop=True, inplace=True)
    filtered_df.drop(columns=["tokens"]).to_csv(
        f'{output_dir}/filtered_data.csv', index=True, index_label='index')


def download_about_me(token, download_dir):
    """
    Download the about me dataset from Huggingface:
    https://huggingface.co/datasets/allenai/aboutme
    """
    login(token=token)
    repo_id = "allenai/aboutme" 
    snapshot_download(repo_id=repo_id, repo_type="dataset", local_dir=download_dir)


def compute_fitering_stats(filtered_data_path, output_dir):
    # Load filtered data
    filtered_df = pd.read_csv(filtered_data_path, index_col=0)
    filtered_df["filtered_roles_data"] = filtered_df["filtered_roles_data"].apply(eval)

    # Load role nouns data
    role_nouns_lookup = load_role_nouns_lookup()

    # Use role_nouns_df to add columns to filtered_df, corresponding to 
    # gender (neut/fem/masc) and role noun set (the neutral role noun variant)
    filtered_df["gender"] = filtered_df["filtered_roles_data"].apply(
        lambda roles_data: role_nouns_lookup.loc[roles_data[0][0]]["gender"]
    )
    filtered_df["role_noun_set"] = filtered_df["filtered_roles_data"].apply(
        lambda roles_data: role_nouns_lookup.loc[roles_data[0][0]]["role_noun_set"]
    )

    # Save a table with each role noun set as a row, and columns for neut/masc/fem
    result_df = []
    for curr_role_noun, curr_df in filtered_df[["role_noun_set", "gender"]].groupby("role_noun_set"):
        curr_gender_df = curr_df.groupby("gender").count()
        curr_result =  {
            "role_noun_set": curr_role_noun,
        }
        for gender in ["neutral", "masculine", "feminine"]:
            if gender in curr_gender_df.index:
                curr_result[gender] = curr_gender_df.loc[gender]["role_noun_set"]
            else:
                curr_result[gender] = 0
        result_df.append(curr_result)
    result_df =  pd.DataFrame(result_df)
    result_df.to_csv(f"{output_dir}/filtered_role_noun_counts.csv")

    # Create visualizations for each role noun set, with neut/masc/fem as columns
    for _, row in result_df.iterrows():
        labels = ["neutral", "masculine", "feminine"]
        values = [row[label] for label in labels]

        fig, ax = plt.subplots()
        ax.bar(labels, values, label=labels)
        ax.set_ylabel('Number of items')
        ax.set_title(f"{row['role_noun_set']} counts")
        plt.savefig(f"{output_dir}/filtered_role_noun_counts_{row['role_noun_set']}.png")


def sample_sentences(filtered_data_path, output_dir, n=6):
    # Load filtered data
    filtered_df = pd.read_csv(filtered_data_path, index_col=0)
    filtered_df["filtered_roles_data"] = filtered_df["filtered_roles_data"].apply(eval)

    # Load role nouns data
    role_nouns_lookup = load_role_nouns_lookup()

    # Use role_nouns_df to add columns to filtered_df, corresponding to 
    # gender (neut/fem/masc) and role noun set (the neutral role noun variant)
    filtered_df["gender"] = filtered_df["filtered_roles_data"].apply(
        lambda roles_data: role_nouns_lookup.loc[roles_data[0][0]]["gender"]
    )
    filtered_df["role_noun_set"] = filtered_df["filtered_roles_data"].apply(
        lambda roles_data: role_nouns_lookup.loc[roles_data[0][0]]["role_noun_set"]
    )
    filtered_df["role_noun"] = filtered_df["filtered_roles_data"].apply(
        lambda roles_data: roles_data[0][0]
    )

    # Sample N sentences per role noun set
    result_df = []
    for _, curr_df in filtered_df.groupby("role_noun"):
        sample_size = min(len(curr_df), n)
        result_df.append(curr_df.sample(sample_size))
    result_df = pd.concat(result_df)

    # Process sentences
    def process_sentence(row):
        sentence = row["sentence"]

        # (a) remove "(EMT)"
        sentence = sentence.replace(" (EMT)", "")

        # (b) replace variants with lowercase versions
        _, curr_start, curr_end = row["filtered_roles_data"][0]
        sentence = sentence[:curr_start] + sentence[curr_start:curr_end].lower() + sentence[curr_end:]

        return sentence


    result_df.to_csv(f"{output_dir}/role_noun_sample_nov25.csv")




if __name__=="__main__":
    # Download the raw About Me dataset
    # huggingface_token = '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'
    download_dir = '/ais/hal9000/datasets/AboutMe/about-me'
    # download_about_me(huggingface_token, download_dir)

    about_zipped_dir = f'{download_dir}/about_pages_zipped'
    individuals_dir = f'{download_dir}/individuals_pages'
    about_page_meta_path = f'{about_zipped_dir}/about_pages_meta.json.gz'

    # Retrieve the hostnames corresponding to individuals
    # individuals_hostnames = get_individuals_hostnames(about_page_meta_path)

    # Unzip and save the data as a CSV file
    # about_pages_paths = [f'{about_zipped_dir}/about_pages-{i}.json.gz' for i in range(14)]
    # save_data_to_csv(about_pages_paths, individuals_dir, individuals_hostnames)

    # Filter the CSV file for relevant rows
    # filter_csv(f'{individuals_dir}/data_with_roles.csv', individuals_dir)

    # CURRENT VERSION - added a couple more role noun sets (substring cases); 
    # filter out names from US Social Security dataset
    # We start with 5721074 rows.
    # After filtering for the specific role nouns: we have 133350 rows.
    # After filtering for first person perspective: we have 47698 rows.
    # After filtering to remove quotes: we have 45063 rows.
    # After filtering for proper nouns and gendered terms: we have 23040 rows.
    # After filtering to remove duplicate sentences: we have 21206 rows.
    # After filtering for length (11-35 words): we have 16220 rows.

    # INTERMEDIATE VERSION - filters out PEOPLE entities using spacy NER + sentences containing the word "name"
    # We start with 5721074 rows.
    # After filtering for the specific role nouns: we have 106467 rows.
    # After filtering for first person perspective: we have 28303 rows.
    # After filtering to remove quotes: we have 26442 rows.
    # After filtering for proper nouns and gendered terms: we have 18638 rows.
    # After filtering to remove duplicate sentences: we have 17255 rows.
    # After filtering for length (max 20 words): we have 6001 rows.

    # PREVIOUS VERSION - filters out ALL PROPER NOUNS 
    # (also only considered \" quotes but not the asymmetrical double quotes)
    # We start with 5721074 rows.
    # After filtering for the specific role nouns: we have 106467 rows.
    # After filtering for first person perspective: we have 28303 rows.
    # After filtering to remove quotes: we have 27584 rows.
    # After filtering for proper nouns and gendered terms: we have 7657 rows.
    # After filtering for length (max 20 words): we have 3578 rows.


    # Compute stats + make visualizations based on filtered data
    # compute_fitering_stats(f'{individuals_dir}/filtered_data.csv', "random_scripts/Nov25")

    # Role noun sets with > 5 occurrences of each variant (current version - Nov 25)
    #     Unnamed: 0    role_noun_set  neutral  masculine  feminine
    # 5            5   businessperson       25        268       163
    # 6            6  camera operator      254        249        13
    # 8            8      chairperson      161        687        15
    # 13          13     craftsperson       51        372        23
    # 18          18              fan     7232         41        94
    # 28          28           maniac       82         17         7
    # 38          38      salesperson      345        371        28
    # 40          40           server      146        102       267
    # 44          44     spokesperson      215         39         5

    sample_sentences(f'{individuals_dir}/filtered_data.csv', "data")