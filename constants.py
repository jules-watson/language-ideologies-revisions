"""
Constants to be used within the pipeline.

Author: Raymond Liu
Date: May 2024
"""

TOKENS_PER_MINUTE_LIMIT = 300000
REQUESTS_PER_MINUTE_LIMIT = 10000

# MODEL_NAME = "gpt-4o"
MODEL_NAME = "llama-3.1-8B-Instruct"
# MODEL_NAME = "gemma-2-9b-it"
# MODEL_NAME = "Mistral-Nemo-Instruct-2407"

USE_SHARDS = False
# USE_SHARDS = True

# FOR context_social_gender analysis (non-Instruct model)
# MODEL_NAME = "llama-3.1-8B"

MODEL_NAMES = ["gpt-4o", "llama-3.1-8B-Instruct", "gemma-2-9b-it", "Mistral-Nemo-Instruct-2407"]

# EXPERIMENT_PATH = "analyses/piloting_pronouns_genders"
EXPERIMENT_PATH = "analyses/full_revise_if_needed"

# EXPERIMENT_PATH = "context_social_gender/analyses/piloting_pronouns_genders"
# EXPERIMENT_PATH = "context_social_gender/analyses/full_sample"


GENDER_INFORMATION_CONDITIONS = {
    'gender declaration man',
    'gender declaration nonbinary',
    'gender declaration woman',
    'pronoun declaration they/them',
    'pronoun declaration she/her',
    'pronoun declaration he/him',
    'pronoun usage their',
    'pronoun usage her',
    'pronoun usage his',
}
