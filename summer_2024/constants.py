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
MODEL_NAME = "llama-3.1-8B-Instruct"

# MODEL_NAMES = ["gpt-3.5-turbo", 'gpt-4-turbo', 'gpt-4o']
# MODEL_NAMES = ['gpt-3.5-turbo', 'gpt-4o']
# MODEL_NAMES = ["llama-3.1-8B-Instruct"]

AN_NOUNS = {"anchor", "anchorman", "anchorwoman"}

EXPERIMENT_PATH = "piloting_experiment_3_way"
#EXPERIMENT_PATH = "piloting_if_needed"
#EXPERIMENT_PATH = "piloting_man"
#EXPERIMENT_PATH = "piloting_non_binary"
#EXPERIMENT_PATH = "piloting_woman"
#EXPERIMENT_PATH = "piloting_experiment_2_way"