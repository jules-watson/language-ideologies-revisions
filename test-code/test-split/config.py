stimuli = {
    "noun_male": "Alex has passed many bills while working as a congressman.",
    "noun_female": "Alex has passed many bills while working as a congresswoman.",
    "noun_neutral": "Alex has passed many bills while working as a congressperson.",
    "adj_male": "Alex looked very handsome in the photo.",
    "adj_female": "Alex looked very beautiful in the photo.",
    "adj_neutral": "Alex looked very attractive in the photo."
}

n_responses = 3
MAX_TOKENS = 500

# models = ["gpt-3.5-turbo", "gpt-4-turbo", "gpt-4o"]
models = ["gpt-3.5-turbo"]

prompts = {
    "short": "Improve the following sentence and explain the changes made:",
    "long": """Improve the following sentence and explain the changes made. 
Label the improved sentence with "Sentence" and the explanation with "Explanation"."""
}