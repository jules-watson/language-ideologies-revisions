"""
Check if data/papineau_role_nouns.json matches the raw Papineau role nouns data,
by transforming the raw data into the format matching the new format.
"""

import json

raw_papineau_path = "random_scripts/papineau_role_nouns_raw.json"
processed_papineau_path = "data/papineau_role_nouns.json"

with open(raw_papineau_path, 'r') as raw_file, open(processed_papineau_path, 'r') as processed_file:
    raw, processed = json.load(raw_file), json.load(processed_file)

    transformed_raw = {
        "neutral": {},
        "feminine": {},
        "masculine": {}
    }
    for role_noun_triplet in raw:
        transformed_raw["neutral"][role_noun_triplet[0]] = role_noun_triplet[0]
        transformed_raw["masculine"][role_noun_triplet[0]] = role_noun_triplet[1]
        transformed_raw["feminine"][role_noun_triplet[0]] = role_noun_triplet[2]
    
    # They should be the same!
    if processed == transformed_raw:
        print("They're the same!")
    else:
        print("Fix the error")