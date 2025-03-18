"""
Constants to be used within the pipeline.

Author: Raymond Liu
Date: May 2024
"""

TOKENS_PER_MINUTE_LIMIT = 300000
REQUESTS_PER_MINUTE_LIMIT = 10000

#MODEL_NAME = "gpt-3.5-turbo"
#MODEL_NAME = "gpt-4-turbo"
# MODEL_NAME = "gpt-4o"
# MODEL_NAME = "llama-3.1-8B-Instruct"
MODEL_NAME = "gemma-2-9b-it"

# FOR context_social_gender analysis (non-Instruct model)
# MODEL_NAME = "llama-3.1-8B"

# MODEL_NAMES = ["gpt-3.5-turbo", 'gpt-4-turbo', 'gpt-4o']
# MODEL_NAMES = ['gpt-3.5-turbo', 'gpt-4o']
MODEL_NAMES = ["gpt-4o", "llama-3.1-8B-Instruct", "gemma-2-9b-it"]

# AN_NOUNS = {"anchor", "anchorman", "anchorwoman"}

#EXPERIMENT_PATH = "piloting_experiment_3_way"
#EXPERIMENT_PATH = "piloting_if_needed"
#EXPERIMENT_PATH = "piloting_man"
#EXPERIMENT_PATH = "piloting_non_binary"
#EXPERIMENT_PATH = "piloting_woman"
#EXPERIMENT_PATH = "piloting_experiment_2_way"
# EXPERIMENT_PATH = "piloting_pronouns_genders"
# EXPERIMENT_PATH = "piloting_pronouns_genders_if_applicable_wording"
# EXPERIMENT_PATH =  "piloting_pronouns_genders_please_revise_prompt_wording"
# EXPERIMENT_PATH =  "piloting_pronouns_genders_please_revise_if_needed_prompt_wording"

# EXPERIMENT_PATH = "analyses/piloting_jan1/improve"
# EXPERIMENT_PATH = "analyses/piloting_jan1/improve_if_needed"
# EXPERIMENT_PATH = "analyses/piloting_jan1/revise"
# EXPERIMENT_PATH = "analyses/piloting_jan1/revise_if_needed"

# EXPERIMENT_PATH = "context_social_gender/analyses/piloting_pronouns_genders"

EXPERIMENT_PATH = "analyses/full_revise_if_needed"

